#include "stop.hpp"
#include <iostream>
#include <stdexcept>

static void check(const std::vector<std::string> & chunks, const std::vector<std::string> & stops,
                  const std::string & expected, bool matched) {
    std::string raw, emitted;
    bool stopped=false;
    for(const auto & chunk:chunks) {
        raw+=chunk;
        const auto boundary=ring::stop_boundary(raw,stops);
        const auto visible=raw.substr(0,boundary.visible);
        if(visible.compare(0,emitted.size(),emitted)!=0) throw std::runtime_error("Retracted streamed content");
        emitted=visible;
        if(boundary.matched) {stopped=true;break;}
    }
    // EOF/length flushes an unmatched prefix rather than silently dropping it.
    if(!stopped) emitted=raw;
    if(emitted!=expected || stopped!=matched) throw std::runtime_error("Incorrect stop boundary");
}

int main() {
    check({"hello E","N","D trailing"},{"END"},"hello ",true);
    check({"E","N","D"},{"END"},"",true);
    check({"hello E","N"},{"END"},"hello EN",false);
    check({"hello E","no stop"},{"END"},"hello Eno stop",false);
    check({"alpha STOP trailing END"},{"END","STOP"},"alpha ",true);
    check({"a","b","c"},{"abc","b"},"a",true);
    check({"abc"},{"","xyz"},"abc",false);
    check({"Привет К","ОН","ЕЦ!"},{"КОНЕЦ"},"Привет ",true);
    check({"hello\n","\n tail"},{"\n\n","\nEND"},"hello",true);
    std::cout<<"9 streaming stop cases passed\n";
}
