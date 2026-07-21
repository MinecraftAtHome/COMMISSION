#include "shroomposter.h"
#include "config.h"

#if defined(_WIN32) || __has_include(<openssl/err.h>)
#include "cpp20_http_client.hpp"
#define SHROOMPOSTER_HTTP_AVAILABLE 1
#else
#define SHROOMPOSTER_HTTP_AVAILABLE 0
#endif

#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <string>
#include <exception>

static std::mutex g_mutex;
static std::condition_variable g_cv;

static std::queue<PostResult> g_queue;

static std::thread g_thread;
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_http_unavailable_logged{false};

static void shroomposter_worker()
{
    while (true)
    {
        PostResult result;

        {
            std::unique_lock lock(g_mutex);

            g_cv.wait(lock, [] {
                return !g_queue.empty() || !g_running;
            });

            if (!g_running && g_queue.empty())
                break;

            result = g_queue.front();
            g_queue.pop();
        }

        std::string json =
            "{\"data\":[{"
            "\"seed\":" + std::to_string(result.seed) +
            ",\"x\":" + std::to_string(result.x) +
            ",\"z\":" + std::to_string(result.z) +
            ",\"claimed_size\":" + std::to_string(result.claimed_size) +
            "}]}";

        std::printf("JSON: %s\n", json.c_str());

#ifdef LARGE_BIOMES
        constexpr char endpoint[] =
            "https://shroomweb.0xa.pw/large_biomes";
#else
        constexpr char endpoint[] =
            "https://shroomweb.0xa.pw/small_biomes";
#endif

#if SHROOMPOSTER_HTTP_AVAILABLE
        try
        {
            auto response =
                http_client::post(
                    endpoint,
                    http_client::Protocol::Https
                )
                .add_header({
                    .name = "api-key",
                    .value = shroomin_api_key
                })
                .add_header({
                    .name = "Content-Type",
                    .value = "application/json"
                })
                .set_body(json)
                .send();

            std::printf(
                "POST %" PRIi64 " -> HTTP %d %.*s\n",
                result.seed,
                static_cast<int>(response.get_status_code()),
                (int)response.get_status_message().size(),
                response.get_status_message().data()
            );

            auto body = response.get_body_string();

            if (!body.empty())
            {
                std::printf(
                    "Response: %.*s\n",
                    (int)body.size(),
                    body.data()
                );
            }
        }
        catch (const std::exception& e)
        {
            std::fprintf(
                stderr,
                "POST failed: %s\n",
                e.what()
            );
        }
#else
        if (!g_http_unavailable_logged.exchange(true)) {
            std::fprintf(stderr, "HTTP posting disabled: OpenSSL headers not found at build time\n");
        }
#endif
    }
}

void shroomposter_start()
{
    std::printf(
        "API KEY = [%s]\n",
        shroomin_api_key.c_str()
    );

    g_running = true;
    g_thread = std::thread(shroomposter_worker);
}

void shroomposter_stop()
{
    g_running = false;
    g_cv.notify_all();

    if (g_thread.joinable())
        g_thread.join();
}

void shroomposter_submit(
    int64_t seed,
    int32_t x,
    int32_t z,
    int64_t claimed_size)
{
    {
        std::lock_guard lock(g_mutex);

        g_queue.push({
            seed,
            x,
            z,
            claimed_size
        });
    }

    g_cv.notify_one();
}
