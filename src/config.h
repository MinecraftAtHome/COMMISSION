#pragma once

#include <string>
#include <unordered_map>
#include <stdexcept>

extern std::unordered_map<std::string, long long> config;
extern std::string version;
extern std::string shroomin_api_key;

void loadConfig();

inline long long cfg(const std::string& key, long long defaultValue)
{
    auto it = config.find(key);
    return it == config.end() ? defaultValue : it->second;
}