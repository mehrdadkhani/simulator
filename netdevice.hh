/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef NETDEVICE_HH
#define NETDEVICE_HH

#include <string>
#include <functional>
#include <netinet/in.h>

#include "file_descriptor.hh"

void interface_ioctl( FileDescriptor & fd, const int request,
                      const std::string & name,
                      std::function<void( ifreq &ifr )> ifr_adjustment);

class TunDevice
{
private:
    FileDescriptor fd_;
public:
    TunDevice( const std::string & name, const std::string & addr, const std::string & dstaddr );

    FileDescriptor & fd( void ) { return fd_; }
};

class VirtualEthernetPair
{
private:
    std::string name;

public:
    VirtualEthernetPair( const std::string & s_outside_name, const std::string & s_inside_name );
    ~VirtualEthernetPair();
};

#endif /* NETDEVICE_HH */
