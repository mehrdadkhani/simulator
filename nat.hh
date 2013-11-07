/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef NAT_HH
#define NAT_HH

/* Network Address Translator */

#include <string>
#include <unistd.h>

#include "config.h"
#include "system_runner.hh"
#include "address.hh"

/* RAII class to make connections coming from the ingress address
   look like they're coming from the output device's address.

   We mark the connections on entry from the ingress address (with our PID),
   and then look for the mark on output. */

class NAT
{
private:
    class Rule {
    private:
        std::vector< std::string > arguments;

    public:
        Rule( const std::vector< std::string > & s_args )
            : arguments( s_args )
        {
            std::vector< std::string > command = { IPTABLES, "-t", "nat", "-A" };
            command.insert( command.end(), arguments.begin(), arguments.end() );
            run( command );
        }

        ~Rule()
        {
            std::vector< std::string > command = { IPTABLES, "-t", "nat", "-D" };
            command.insert( command.end(), arguments.begin(), arguments.end() );
            run( command );
        }

        Rule( const Rule & other ) = delete;
        const Rule & operator=( const Rule & other ) = delete;
    };

    Rule pre_, post_;

public:
    NAT( const Address & ingress_addr )
    : pre_( { "PREROUTING", "-s", ingress_addr.ip(), "-j", "CONNMARK", "--set-mark", std::to_string( getpid() ) } ),
      post_( { "POSTROUTING", "-j", "MASQUERADE", "-m", "connmark", "--mark", std::to_string( getpid() ) } )
    {}
};

#endif /* NAT_HH */
