#include <libstemmer.h>
#include <stdio.h>
#include <string.h>

static int check_stem(const char *language, const char *word,
                      const char *expected) {
  struct sb_stemmer *stemmer = sb_stemmer_new(language, "UTF_8");
  const sb_symbol *result;
  int matches;
  if (!stemmer) {
    fprintf(stderr, "Cannot create %s stemmer\n", language);
    return 1;
  }
  result = sb_stemmer_stem(stemmer, (const sb_symbol *)word, (int)strlen(word));
  matches = result && sb_stemmer_length(stemmer) == (int)strlen(expected) &&
            memcmp(result, expected, strlen(expected)) == 0;
  if (!matches) {
    fprintf(stderr, "%s: unexpected stem for %s\n", language, word);
  }
  sb_stemmer_delete(stemmer);
  return !matches;
}

int main(void) {
  const char **language = sb_stemmer_list();
  int count = 0;
  int failures = 0;
  for (; *language; ++language) {
    struct sb_stemmer *stemmer = sb_stemmer_new(*language, "UTF_8");
    if (!stemmer) {
      fprintf(stderr, "Cannot create %s stemmer\n", *language);
      return 1;
    }
    if (!sb_stemmer_stem(stemmer, (const sb_symbol *)"testing", 7)) {
      fprintf(stderr, "Cannot run %s stemmer\n", *language);
      failures++;
    }
    sb_stemmer_delete(stemmer);
    count++;
  }
  failures += check_stem("english", "running", "run");
  failures += check_stem("english", "skies", "sky");
  failures += check_stem("english", "relational", "relat");
  /* UTF-8 Arabic: the definite article is removed from "the book". */
  failures +=
      check_stem("arabic", "\xd8\xa7\xd9\x84\xd9\x83\xd8\xaa\xd8\xa7\xd8\xa8",
                 "\xd9\x83\xd8\xaa\xd8\xa7\xd8\xa8");
  if (count < 36) {
    fprintf(stderr, "Expected at least 36 Snowball 3.1.1 stemmers, found %d\n",
            count);
    return 1;
  }
  printf("Validated %d UTF-8 stemmers and 4 expected stems\n", count);
  return failures ? 1 : 0;
}
