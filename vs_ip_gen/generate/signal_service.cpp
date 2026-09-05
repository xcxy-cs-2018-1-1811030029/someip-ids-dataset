// signal_service.cpp
// SOME/IP publish-subscribe service for IDS dataset generation.
// Based on vsomeip notify-sample. Adds:
//   - logging of each sent notification to a CSV (ts,event,payload_hex)
//   - attack modes: normal / dos / fuzz / slowslow / replay
//
// Build (see build_samples.sh). Run:
//   env VSOMEIP_CONFIGURATION=../../config/vsomeip-local.json \
//       VSOMEIP_APPLICATION_NAME=service-sample ./signal_service --cycle 1000 --out svc.csv

#ifndef VSOMEIP_ENABLE_SIGNAL_HANDLING
#include <csignal>
#if defined(__linux__) || defined(__QNX__)
#include <pthread.h>
#endif
#endif
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <mutex>
#include <fstream>
#include <random>
#include <cstring>

#include <vsomeip/vsomeip.hpp>

#define SVC          0x1234
#define INST         0x5678
#define EVENT        0x8778
#define EVENTGROUP   0x4465
#define GET_METHOD   0x0001
#define SET_METHOD   0x0002

static std::string hexstr(const uint8_t* d, uint32_t n) {
    std::string h; char buf[8];
    for (uint32_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof(buf), "%02X", d[i]); h += buf;
        if (i + 1 < n) h += ' ';
    }
    return h;
}

static double now_ts() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

class service_sample {
public:
    service_sample(uint32_t _cycle, std::string _out, std::string _attack, float _intensity = 1.0f,
                   float _noise = 0.0f)
        : app_(vsomeip::runtime::get()->create_application()), is_registered_(false),
          cycle_(_cycle), out_(_out), attack_(_attack), intensity_(_intensity), noise_(_noise),
          blocked_(false), running_(true),
          is_offered_(false),
          offer_thread_(std::bind(&service_sample::run, this)),
          notify_thread_(std::bind(&service_sample::notify, this)) {
        if (!out_.empty()) log_.open(out_, std::ios::out | std::ios::trunc);
        if (log_.is_open()) log_ << "ts,event,payload_len,payload_hex\n";
    }

    bool init() {
        std::scoped_lock its_lock(mutex_);
        if (!app_->init()) { std::cerr << "Couldn't initialize application" << std::endl; return false; }
        app_->register_state_handler(std::bind(&service_sample::on_state, this, std::placeholders::_1));
        app_->register_message_handler(SVC, INST, GET_METHOD,
                                       std::bind(&service_sample::on_get, this, std::placeholders::_1));
        app_->register_message_handler(SVC, INST, SET_METHOD,
                                       std::bind(&service_sample::on_set, this, std::placeholders::_1));
        std::set<vsomeip::eventgroup_t> its_groups;
        its_groups.insert(EVENTGROUP);
        app_->offer_event(SVC, INST, EVENT, its_groups, vsomeip::event_type_e::ET_FIELD,
                          std::chrono::milliseconds::zero(), false, true, nullptr,
                          vsomeip::reliability_type_e::RT_UNKNOWN);
        { std::scoped_lock its_lock(payload_mutex_); payload_ = vsomeip::runtime::get()->create_payload(); }
        blocked_ = true; condition_.notify_one();
        return true;
    }

    void start() { app_->start(); }

    void stop() {
        running_ = false; blocked_ = true; condition_.notify_one(); notify_condition_.notify_one();
        app_->clear_all_handler(); stop_offer();
        if (std::this_thread::get_id() != offer_thread_.get_id() && offer_thread_.joinable()) offer_thread_.join();
        if (std::this_thread::get_id() != notify_thread_.get_id() && notify_thread_.joinable()) notify_thread_.join();
        app_->stop();
        if (log_.is_open()) log_.close();
    }

    void offer() {
        std::scoped_lock its_lock(notify_mutex_);
        app_->offer_service(SVC, INST); is_offered_ = true; notify_condition_.notify_one();
    }
    void stop_offer() { app_->stop_offer_service(SVC, INST); is_offered_ = false; }

    void on_state(vsomeip::state_type_e _state) {
        if (_state == vsomeip::state_type_e::ST_REGISTERED) is_registered_ = true;
        else is_registered_ = false;
    }

    void on_get(const std::shared_ptr<vsomeip::message>& _m) {
        auto r = vsomeip::runtime::get()->create_response(_m);
        { std::scoped_lock its_lock(payload_mutex_); r->set_payload(payload_); }
        app_->send(r);
    }

    void on_set(const std::shared_ptr<vsomeip::message>& _m) {
        auto r = vsomeip::runtime::get()->create_response(_m);
        { std::scoped_lock its_lock(payload_mutex_); payload_ = _m->get_payload(); r->set_payload(payload_); }
        app_->send(r);
        app_->notify(SVC, INST, EVENT, payload_);
    }

    void run() {
        std::unique_lock its_lock(mutex_);
        condition_.wait(its_lock, [this] { return blocked_; });
        bool is_offer(true);
        while (running_) {
            if (is_offer) offer(); else stop_offer();
            for (int i = 0; i < 10 && running_; i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            is_offer = !is_offer;
        }
    }

    void notify() {
        // realistic 12-byte payload: 3 little-endian floats (sensor signals)
        std::mt19937 rng(12345);
        // sensor noise: separate RNG so attack branches that draw from `rng`
        // (ctx_tamper, fuzz) stay bit-identical with or without --noise
        std::mt19937 noise_rng(777);
        std::normal_distribution<float> nz(0.0f, noise_);
        double val = 0.0;
        uint8_t base[12];
        while (running_) {
            std::unique_lock its_lock(notify_mutex_);
            notify_condition_.wait(its_lock, [this] { return is_offered_ || !running_; });
            while (is_offered_ && running_) {
                float a = 100.0f + 50.0f * std::sin(static_cast<float>(val) * 0.1f);
                float b = 50.0f + 30.0f * std::cos(static_cast<float>(val) * 0.2f);
                float c = 20.0f + static_cast<float>((static_cast<int>(val)) % 100);
                if (noise_ > 0.0f) {
                    a += nz(noise_rng);
                    b += nz(noise_rng);
                    c += nz(noise_rng);
                }
                std::memcpy(base, &a, 4);
                std::memcpy(base + 4, &b, 4);
                std::memcpy(base + 8, &c, 4);
                uint32_t n = 12;

                if (attack_ == "tamper") {
                    // semantics-preserving tamper: keep structure/timing/byte-stats normal,
                    // but the decoded speed value VARIES while being IMPLAUSIBLE (outside the normal
                    // range), with the magnitude scaled by `intensity`.
                    float bad = 150.0f + intensity_ * 150.0f + 50.0f * std::sin(static_cast<float>(val) * 0.15f);
                    std::memcpy(base, &bad, 4);
                } else if (attack_ == "ctx_tamper") {
                    // contextual tamper: each signal individually stays in its normal range,
                    // but the CROSS-SIGNAL relationship is broken by drawing each signal from an
                    // independent uniform within its normal range (destroys the normal joint/temporal
                    // correlation), while bytes/timing remain plausible.
                    float s = 50.0f + (static_cast<float>(rng() % 1000) / 1000.0f) * 100.0f;   // 50..150
                    float acc = 20.0f + (static_cast<float>(rng() % 1000) / 1000.0f) * 60.0f;  // 20..80
                    float yaw = 20.0f + (static_cast<float>(rng() % 1000) / 1000.0f) * 99.0f;  // 20..119
                    std::memcpy(base, &s, 4); std::memcpy(base + 4, &acc, 4); std::memcpy(base + 8, &yaw, 4);
                } else if (attack_ == "fuzz") {
                    // fuzz complexity scaled by `intensity` (higher -> more malformed bytes).
                    n = 1 + (rng() % (1 + (uint32_t)(intensity_ * 15.0f)));
                    for (uint32_t i = 0; i < n; ++i) base[i] = static_cast<uint8_t>(rng() & 0xFF);
                } else if (attack_ == "replay") {
                    // replay: keep the same bytes, same timing as normal.
                }

                {
                    std::scoped_lock its_lock(payload_mutex_);
                    payload_->set_data(base, n);
                    app_->notify(SVC, INST, EVENT, payload_);
                }
                if (log_.is_open()) {
                    log_ << std::fixed << std::setprecision(6) << now_ts() << ","
                         << EVENT << "," << n << "," << hexstr(base, n) << "\n";
                    log_.flush();
                }

                if (attack_ == "dos") {
                    // burst: high rate scaled by `intensity`; each packet a DISTINCT normal-like payload.
                    std::this_thread::sleep_for(std::chrono::milliseconds(
                        1 + (uint32_t)((1.0f - intensity_) * 9.0f)));
                } else if (attack_ == "slowslow") {
                    std::this_thread::sleep_for(std::chrono::milliseconds((uint32_t)(2000.0f / intensity_)));
                } else if (attack_ == "drop") {
                    // message-drop: periodically withhold notifications -> gaps in the stream
                    std::this_thread::sleep_for(std::chrono::milliseconds(
                        ((static_cast<int>(val) % 7) == 0) ? cycle_ * 6 : cycle_));
                } else {
                    // normal, tamper use the same cycle_ -> identical timing
                    std::this_thread::sleep_for(std::chrono::milliseconds(cycle_));
                }
                val += 1.0;
            }
        }
    }

private:
    std::shared_ptr<vsomeip::application> app_;
    bool is_registered_;
    uint32_t cycle_;
    std::string out_, attack_;
    float intensity_, noise_;
    std::ofstream log_;
    std::mutex mutex_; std::condition_variable condition_; bool blocked_;
    bool running_;
    std::mutex notify_mutex_; std::condition_variable notify_condition_; bool is_offered_;
    std::mutex payload_mutex_; std::shared_ptr<vsomeip::payload> payload_;
    std::thread offer_thread_, notify_thread_;
};

int main(int argc, char** argv) {
    uint32_t cycle = 1000;
    float intensity = 1.0f;
    float noise = 0.0f;
    std::string out = "svc.csv", attack = "normal";
    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--cycle" && i + 1 < argc) { cycle = (uint32_t)std::atoi(argv[++i]); }
        else if (a == "--out" && i + 1 < argc) { out = argv[++i]; }
        else if (a == "--attack" && i + 1 < argc) { attack = argv[++i]; }
        else if (a == "--intensity" && i + 1 < argc) { intensity = (float)atof(argv[++i]); }
        else if (a == "--noise" && i + 1 < argc) { noise = (float)atof(argv[++i]); }
    }
    std::cout << "signal_service attack=" << attack << " cycle=" << cycle
              << " intensity=" << intensity << " noise=" << noise
              << " out=" << out << std::endl;

#ifndef VSOMEIP_ENABLE_SIGNAL_HANDLING
    std::signal(SIGINT, [](int){});
#endif
    service_sample its_sample(cycle, out, attack, intensity, noise);
    if (its_sample.init()) {
        // simple signal handling: block SIGINT and run
        its_sample.start();
        return 0;
    }
    return 1;
}
