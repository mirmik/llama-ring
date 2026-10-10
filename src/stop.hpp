#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace ring {
struct StopBoundary {
    size_t visible;
    bool matched;
};

// Withhold incomplete stop prefixes across token boundaries. On a complete
// match the caller truncates both text and its token-aware parser input.
inline StopBoundary stop_boundary(const std::string & text, const std::vector<std::string> & stops) {
    StopBoundary result{text.size(), false};
    for(const auto & stop:stops) {
        if(stop.empty()) continue;
        const auto at=text.find(stop);
        if(at!=std::string::npos) {result.visible=std::min(result.visible,at);result.matched=true;}
    }
    if(result.matched) return result;
    for(const auto & stop:stops) for(size_t n=1;n<stop.size() && n<=text.size();++n)
        if(text.compare(text.size()-n,n,stop,0,n)==0) result.visible=std::min(result.visible,text.size()-n);
    return result;
}
}
