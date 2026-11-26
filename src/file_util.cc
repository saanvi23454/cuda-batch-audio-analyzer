#include "file_util.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <fstream>

namespace audio {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

bool MakeOneDirectory(const std::string& path) {
#ifdef _WIN32
  const int rc = mkdir(path.c_str());
#else
  const int rc = mkdir(path.c_str(), 0755);
#endif
  return rc == 0 || errno == EEXIST;
}

}  // namespace

std::vector<std::string> ListFiles(const std::string& directory,
                                   const std::string& extension) {
  std::vector<std::string> names;
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) return names;
  const std::string wanted = ToLower(extension);
  while (dirent* entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name.size() <= wanted.size()) continue;
    if (ToLower(name.substr(name.size() - wanted.size())) == wanted) {
      names.push_back(name);
    }
  }
  closedir(dir);
  std::sort(names.begin(), names.end());
  return names;
}

bool MakeDirectories(const std::string& path) {
  if (path.empty()) return true;
  for (size_t pos = path.find('/', 1); pos != std::string::npos;
       pos = path.find('/', pos + 1)) {
    if (!MakeOneDirectory(path.substr(0, pos))) return false;
  }
  return MakeOneDirectory(path);
}

std::string JoinPath(const std::string& directory, const std::string& name) {
  if (directory.empty()) return name;
  const char last = directory.back();
  if (last == '/' || last == '\\') return directory + name;
  return directory + "/" + name;
}

std::string StripExtension(const std::string& name) {
  const size_t dot = name.find_last_of('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

bool WritePgm(const std::string& path, int width, int height,
              const std::vector<uint8_t>& pixels) {
  if (pixels.size() != static_cast<size_t>(width) * height) return false;
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  file << "P5\n" << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<const char*>(pixels.data()),
             static_cast<std::streamsize>(pixels.size()));
  return static_cast<bool>(file);
}

}  // namespace audio
