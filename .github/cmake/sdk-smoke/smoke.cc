#include <iostream>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/db/status.h>

int main() {
  const char *message = zvec::GetDefaultMessage(zvec::StatusCode::OK);
  if (!message) {
    std::cerr << "cpp_smoke: GetDefaultMessage() returned null" << std::endl;
    return 1;
  }
  const auto io_backend_type = zvec::ailego::current_io_backend_type();
  const std::string io_backend_description =
      zvec::ailego::current_io_backend_description();
  if (io_backend_description.empty()) {
    std::cerr << "cpp_smoke: current_io_backend_description() returned empty"
              << std::endl;
    return 1;
  }
  std::cout << "cpp_smoke: StatusCode::OK -> " << message << std::endl;
  std::cout << "cpp_smoke: I/O backend " << static_cast<int>(io_backend_type)
            << " -> " << io_backend_description << std::endl;
  std::cout << "cpp_smoke: OK" << std::endl;
  return 0;
}
