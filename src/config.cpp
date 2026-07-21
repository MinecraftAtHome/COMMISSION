#include "config.h"

#include <fstream>
#include <string>

std::unordered_map<std::string, long long> config;
std::string version;
std::string shroomin_api_key;

void loadConfig()
{
    std::ifstream file("config.ini");
    if (!file)
        return;

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        auto pos = line.find('=');

        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        if (key == "version")
        {
            version = value;
            continue;
        }
        
        if (key == "shroomin_api_key")
        {
            shroomin_api_key = value;
            continue;
        }

        try
        {
            config[key] = std::stoll(value);
        }
        catch (...)
        {
        }
    }
}