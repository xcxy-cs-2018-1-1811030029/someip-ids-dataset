// signal_client.cpp
// SOME/IP publish-subscribe client for IDS dataset generation.
// Based on vsomeip subscribe-sample. Adds logging of each received notification
// to a CSV (ts,event,session,payload_len,payload_hex).
//
// Run:
//   env VSOMEIP_CONFIGURATION=../../config/vsomeip-local.json \
//       VSOMEIP_APPLICATION_NAME=client-sample ./signal_client --out cli.csv

#include <csignal>
#if defined(__linux__) || defined(__QNX__)
#include <pthread.h>
#endif
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <fstream>
#include <cstdio>

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

class client_sample {
public:
    client_sample(std::string _out) : app_(vsomeip::runtime::get()->create_application()), out_(_out) {
        if (!out_.empty()) log_.open(out_, std::ios::out | std::ios::trunc);
        if (log_.is_open()) log_ << "ts,event,session,payload_len,payload_hex\n";
    }

    bool init() {
        if (!app_->init()) { std::cerr << "Couldn't initialize application" << std::endl; return false; }
        std::cout << "Client settings [protocol=UDP]" << std::endl;
        app_->register_state_handler(std::bind(&client_sample::on_state, this, std::placeholders::_1));
        app_->register_message_handler(vsomeip::ANY_SERVICE, INST, vsomeip::ANY_METHOD,
                                       std::bind(&client_sample::on_message, this, std::placeholders::_1));
        app_->register_availability_handler(SVC, INST,
            std::bind(&client_sample::on_availability, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        std::set<vsomeip::eventgroup_t> its_groups;
        its_groups.insert(EVENTGROUP);
        app_->request_event(SVC, INST, EVENT, its_groups, vsomeip::event_type_e::ET_FIELD);
        app_->subscribe(SVC, INST, EVENTGROUP);
        return true;
    }

    void start() { app_->start(); }

    void stop() {
        app_->clear_all_handler();
        app_->unsubscribe(SVC, INST, EVENTGROUP);
        app_->release_event(SVC, INST, EVENT);
        app_->stop();
        if (log_.is_open()) log_.close();
    }

    void on_state(vsomeip::state_type_e _state) {
        if (_state == vsomeip::state_type_e::ST_REGISTERED) app_->request_service(SVC, INST);
    }

    void on_availability(vsomeip::service_t _s, vsomeip::instance_t _i, bool _av) {
        std::cout << "Service [" << std::hex << std::setfill('0') << std::setw(4) << _s << "." << _i
                  << "] is " << (_av ? "available." : "NOT available.") << std::endl;
    }

    void on_message(const std::shared_ptr<vsomeip::message>& _r) {
        std::shared_ptr<vsomeip::payload> its_payload = _r->get_payload();
        const uint8_t* d = its_payload->get_data();
        uint32_t n = its_payload->get_length();
        if (log_.is_open()) {
            log_ << std::fixed << std::setprecision(6) << now_ts() << ","
                 << _r->get_method() << "," << std::hex << std::setfill('0') << std::setw(4)
                 << _r->get_session() << std::dec << "," << n << "," << hexstr(d, n) << "\n";
            log_.flush();
        }
        std::cout << "Received Event [" << std::hex << std::setfill('0') << std::setw(4) << _r->get_service()
                  << "." << std::setw(4) << _r->get_instance() << "." << std::setw(4) << _r->get_method()
                  << "] = (" << std::dec << n << ") " << hexstr(d, n) << std::endl;
    }

private:
    std::shared_ptr<vsomeip::application> app_;
    std::string out_;
    std::ofstream log_;
};

int main(int argc, char** argv) {
    std::string out = "cli.csv";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) out = argv[++i];
    }
    client_sample its_sample(out);
    if (its_sample.init()) {
        its_sample.start();
        return 0;
    }
    return 1;
}
