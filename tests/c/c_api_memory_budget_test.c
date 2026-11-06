// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zvec/c_api.h>

#define MIB ((size_t)1024 * 1024)

int main(void) {
  zvec_config_data_t *config = zvec_config_data_create();
  assert(config);
  assert(zvec_config_data_set_memory_limit(config, 128 * MIB) == ZVEC_OK);
  assert(zvec_config_data_set_query_thread_count(config, 2) == ZVEC_OK);
  assert(zvec_config_data_set_optimize_thread_count(config, 1) == ZVEC_OK);
  assert(zvec_initialize(config) == ZVEC_OK);
  zvec_config_data_destroy(config);

  zvec_collection_schema_t *schema = zvec_collection_schema_create("budget");
  zvec_field_schema_t *field =
      zvec_field_schema_create("payload", ZVEC_DATA_TYPE_STRING, false, 0);
  assert(zvec_collection_schema_add_field(schema, field) == ZVEC_OK);
  zvec_field_schema_destroy(field);
  zvec_collection_options_t *options = zvec_collection_options_create();
  assert(zvec_collection_options_set_enable_mmap(options, false) == ZVEC_OK);
  assert(zvec_collection_options_set_max_buffer_size(options, MIB) == ZVEC_OK);
  char path[128];
  snprintf(path, sizeof(path), "zvec_c_memory_%llu_%llu",
           (unsigned long long)time(NULL), (unsigned long long)clock());
  zvec_collection_t *collection = NULL;
  assert(zvec_collection_create_and_open(path, schema, options, &collection) ==
         ZVEC_OK);
  zvec_collection_options_destroy(options);
  zvec_collection_schema_destroy(schema);

  char *payload = (char *)malloc(MIB + 1);
  assert(payload);
  memset(payload, 'x', MIB);
  payload[MIB] = 0;
  zvec_doc_t *doc = zvec_doc_create();
  zvec_doc_set_pk(doc, "one");
  assert(zvec_doc_add_field_by_value(doc, "payload", ZVEC_DATA_TYPE_STRING,
                                     payload, MIB) == ZVEC_OK);
  free(payload);
  const zvec_doc_t *batch[64];
  for (size_t i = 0; i < 64; ++i) batch[i] = doc;
  size_t success = 123, errors = 123;
  assert(zvec_collection_insert(collection, batch, 64, &success, &errors) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);
  assert(success == 0 && errors == 64);
  assert(zvec_collection_update(collection, batch, 64, &success, &errors) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);
  assert(zvec_collection_upsert(collection, batch, 64, &success, &errors) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);

  // The caller owns only a pointer array and one literal. The binding must
  // reject before creating millions of native strings, even when all IDs miss.
  const size_t count = 4000000;
  const char **ids = (const char **)malloc(count * sizeof(char *));
  assert(ids);
  for (size_t i = 0; i < count; ++i) ids[i] = "missing";
  zvec_doc_t **docs = (zvec_doc_t **)1;
  size_t found = 123;
  assert(zvec_collection_fetch(collection, ids, count, NULL, 0, false, &docs,
                               &found) == ZVEC_ERROR_RESOURCE_EXHAUSTED);
  assert(docs == NULL && found == 0);
  assert(zvec_collection_delete(collection, ids, count, &success, &errors) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);
  free(ids);

  // C strings are copied to native query objects only after admission.
  char *filter = (char *)malloc(24 * MIB + 1);
  assert(filter);
  memset(filter, ' ', 24 * MIB);
  filter[24 * MIB] = 0;
  assert(zvec_collection_delete_by_filter(collection, filter) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);
  field = zvec_field_schema_create("denied", ZVEC_DATA_TYPE_INT64, false, 0);
  assert(field);
  assert(zvec_collection_add_column(collection, field, filter) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);
  zvec_field_schema_destroy(field);
  free(filter);

  // Retaining copies must exhaust a shared active allowance, independent of
  // the original document's lifetime. Both documented free APIs refund it.
  void *copies[64];
  size_t copied = 0;
  for (; copied < 64; ++copied) {
    void *value = (void *)1;
    size_t size = 123;
    zvec_error_code_t result = zvec_doc_get_field_value_copy(
        doc, "payload", ZVEC_DATA_TYPE_STRING, &value, &size);
    if (result == ZVEC_ERROR_RESOURCE_EXHAUSTED) {
      assert(value == NULL && size == 0);
      break;
    }
    assert(result == ZVEC_OK && size == MIB);
    copies[copied] = value;
  }
  assert(copied > 0 && copied < 64);
  zvec_doc_destroy(doc);
  for (size_t i = 0; i < copied; ++i) {
    assert(((char *)copies[i])[0] == 'x' &&
           ((char *)copies[i])[MIB - 1] == 'x');
    if (i % 2)
      zvec_free(copies[i]);
    else
      zvec_free_uint8_array((uint8_t *)copies[i]);
  }

  doc = zvec_doc_create();
  zvec_doc_set_pk(doc, "small");
  assert(zvec_doc_add_field_by_value(doc, "payload", ZVEC_DATA_TYPE_STRING,
                                     "ok", 2) == ZVEC_OK);
  // Reusing the retained high water must still obey a new host notification.
  assert(zvec_set_memory_pressure(ZVEC_MEMORY_PRESSURE_CRITICAL) == ZVEC_OK);
  void *blocked = (void *)1;
  size_t blocked_size = 123;
  assert(zvec_doc_get_field_value_copy(doc, "payload", ZVEC_DATA_TYPE_STRING,
                                       &blocked, &blocked_size) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);
  assert(blocked == NULL && blocked_size == 0);
  assert(zvec_set_memory_pressure(ZVEC_MEMORY_PRESSURE_BACKGROUND) == ZVEC_OK);
  for (size_t i = 0; i < 100; ++i) {
    void *value = NULL;
    size_t size = 0;
    assert(zvec_doc_get_field_value_copy(doc, "payload", ZVEC_DATA_TYPE_STRING,
                                         &value, &size) == ZVEC_OK);
    assert(size == 2 && memcmp(value, "ok", 2) == 0);
    zvec_free(value);
  }
  assert(zvec_set_memory_pressure(ZVEC_MEMORY_PRESSURE_NORMAL) == ZVEC_OK);
  const char *pk = zvec_doc_get_pk_copy(doc);
  assert(pk && strcmp(pk, "small") == 0);
  zvec_free((void *)pk);
  uint8_t *serialized = NULL;
  size_t size = 0;
  assert(zvec_doc_serialize(doc, &serialized, &size) == ZVEC_OK && size > 0);
  assert(zvec_set_memory_pressure(ZVEC_MEMORY_PRESSURE_CRITICAL) == ZVEC_OK);
  zvec_doc_t *decoded = (zvec_doc_t *)1;
  assert(zvec_doc_deserialize(serialized, size, &decoded) ==
         ZVEC_ERROR_RESOURCE_EXHAUSTED);
  assert(decoded == NULL);
  assert(zvec_set_memory_pressure(ZVEC_MEMORY_PRESSURE_NORMAL) == ZVEC_OK);
  assert(zvec_doc_deserialize(serialized, size, &decoded) == ZVEC_OK);
  assert(decoded);
  zvec_free_uint8_array(serialized);
  pk = zvec_doc_get_pk_copy(decoded);
  assert(pk && strcmp(pk, "small") == 0);
  zvec_free((void *)pk);
  zvec_doc_destroy(decoded);
  char **names = NULL;
  size_t names_count = 0;
  assert(zvec_doc_get_field_names(doc, &names, &names_count) == ZVEC_OK);
  assert(names_count == 1 && strcmp(names[0], "payload") == 0);
  zvec_free_str_array(names, names_count);
  batch[0] = doc;
  assert(zvec_collection_insert(collection, batch, 1, &success, &errors) ==
         ZVEC_OK);
  assert(success == 1 && errors == 0);
  zvec_doc_destroy(doc);
  const char *small_id = "small";
  docs = NULL;
  found = 0;
  assert(zvec_collection_fetch(collection, &small_id, 1, NULL, 0, false,
                               &docs, &found) == ZVEC_OK);
  assert(found == 1 && docs);
  assert(zvec_collection_destroy(collection) == ZVEC_OK);
  assert(zvec_collection_close(collection) == ZVEC_OK);
  // Returned Docs and extracted buffers remain usable after collection close.
  void *value = NULL;
  size_t value_size = 0;
  assert(zvec_doc_get_field_value_copy(docs[0], "payload", ZVEC_DATA_TYPE_STRING,
                                       &value, &value_size) == ZVEC_OK);
  assert(value_size == 2 && memcmp(value, "ok", 2) == 0);
  zvec_docs_free(docs, found);
  zvec_free(value);
  puts("C API memory admission and recovery passed");
  return 0;
}
