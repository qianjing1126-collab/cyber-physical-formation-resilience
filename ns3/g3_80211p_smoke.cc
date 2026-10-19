#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <iostream>

using namespace ns3;

int
main(int argc, char* argv[])
{
    CommandLine command(__FILE__);
    command.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(2);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("OfdmRate6MbpsBW10MHz"),
                                 "ControlMode",
                                 StringValue("OfdmRate6MbpsBW10MHz"));
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    Simulator::Stop(MilliSeconds(1));
    Simulator::Run();
    Simulator::Destroy();

    std::cout << "G3_80211P_SMOKE_OK nodes=" << nodes.GetN()
              << " devices=" << devices.GetN() << std::endl;
    return 0;
}
