#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ns3;

namespace
{

struct PacketMeta
{
    uint32_t sender;
    uint32_t sequence;
    double generationTime;
    uint32_t size;
    bool probe;
};

class TnsmBeaconHeader : public Header
{
  public:
    TnsmBeaconHeader() = default;

    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TnsmBeaconHeader")
                                .SetParent<Header>()
                                .SetGroupName("Network")
                                .AddConstructor<TnsmBeaconHeader>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return 16;
    }

    void Serialize(Buffer::Iterator start) const override
    {
        start.WriteHtonU32(m_sender);
        start.WriteHtonU32(m_sequence);
        start.WriteHtonU64(m_generationNanoseconds);
    }

    uint32_t Deserialize(Buffer::Iterator start) override
    {
        m_sender = start.ReadNtohU32();
        m_sequence = start.ReadNtohU32();
        m_generationNanoseconds = start.ReadNtohU64();
        return GetSerializedSize();
    }

    void Print(std::ostream& stream) const override
    {
        stream << "sender=" << m_sender << " seq=" << m_sequence;
    }

    void Set(uint32_t sender, uint32_t sequence, uint64_t generationNanoseconds)
    {
        m_sender = sender;
        m_sequence = sequence;
        m_generationNanoseconds = generationNanoseconds;
    }

    uint32_t GetSender() const
    {
        return m_sender;
    }

    uint32_t GetSequence() const
    {
        return m_sequence;
    }

    uint64_t GetGenerationNanoseconds() const
    {
        return m_generationNanoseconds;
    }

  private:
    uint32_t m_sender{0};
    uint32_t m_sequence{0};
    uint64_t m_generationNanoseconds{0};
};

struct TxRecord
{
    uint64_t uid;
    PacketMeta meta;
};

struct RxRecord
{
    uint64_t uid;
    uint32_t sender;
    uint32_t receiver;
    uint32_t sequence;
    double generationTime;
    double receptionTime;
    uint32_t size;
};

std::vector<TxRecord> g_txRecords;
std::vector<RxRecord> g_rxRecords;
std::vector<uint32_t> g_sequences;
uint32_t g_probeNodes = 0;
uint64_t g_sendFailures = 0;

uint64_t
LogicalUid(uint32_t sender, uint32_t sequence)
{
    return (static_cast<uint64_t>(sender) << 32) | sequence;
}

void
ReceivePacket(uint32_t receiver, Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from)))
    {
        TnsmBeaconHeader header;
        if (packet->RemoveHeader(header) != header.GetSerializedSize())
        {
            continue;
        }
        const uint32_t sender = header.GetSender();
        if (sender >= g_probeNodes || receiver >= g_probeNodes || sender == receiver)
        {
            continue;
        }
        const uint32_t sequence = header.GetSequence();
        const double generationTime = static_cast<double>(header.GetGenerationNanoseconds()) * 1e-9;
        g_rxRecords.push_back({LogicalUid(sender, sequence),
                               sender,
                               receiver,
                               sequence,
                               generationTime,
                               Simulator::Now().GetSeconds(),
                               packet->GetSize() + header.GetSerializedSize()});
    }
}

void
SendBeaconOnce(uint32_t sender, Ptr<Socket> socket, bool probe, uint32_t packetSize)
{
    const double now = Simulator::Now().GetSeconds();
    const PacketMeta meta{sender, ++g_sequences.at(sender), now, packetSize, probe};
    const uint64_t uid = LogicalUid(sender, meta.sequence);
    TnsmBeaconHeader header;
    header.Set(sender, meta.sequence, Simulator::Now().GetNanoSeconds());
    Ptr<Packet> packet = Create<Packet>(packetSize - header.GetSerializedSize());
    packet->AddHeader(header);
    if (probe)
    {
        g_txRecords.push_back({uid, meta});
    }
    if (socket->Send(packet) < 0)
    {
        ++g_sendFailures;
    }
}

void
WriteOutputs(const std::string& outputDirectory,
             const std::string& protocolVersion,
             const std::string& regime,
             uint64_t seed,
             uint32_t probeNodes,
             uint32_t backgroundNodes,
             double backgroundRate,
             double backgroundRadius,
             uint32_t packetSize,
             double probeRate,
             double warmup,
             double measurement,
             double spacing,
             double pathLossExponent,
             double referenceLoss,
             double txPower,
             double noiseFigure,
             double generationJitterFraction)
{
    std::filesystem::create_directories(outputDirectory);
    std::sort(g_txRecords.begin(), g_txRecords.end(), [](const auto& a, const auto& b) {
        if (a.meta.generationTime != b.meta.generationTime)
        {
            return a.meta.generationTime < b.meta.generationTime;
        }
        if (a.meta.sender != b.meta.sender)
        {
            return a.meta.sender < b.meta.sender;
        }
        return a.meta.sequence < b.meta.sequence;
    });
    std::sort(g_rxRecords.begin(), g_rxRecords.end(), [](const auto& a, const auto& b) {
        if (a.receptionTime != b.receptionTime)
        {
            return a.receptionTime < b.receptionTime;
        }
        if (a.sender != b.sender)
        {
            return a.sender < b.sender;
        }
        if (a.sequence != b.sequence)
        {
            return a.sequence < b.sequence;
        }
        return a.receiver < b.receiver;
    });

    std::ofstream tx(outputDirectory + "/tx.csv");
    tx << "uid,tx_id,seq,t_gen_s,size_bytes\n" << std::setprecision(12);
    for (const auto& row : g_txRecords)
    {
        tx << row.uid << ',' << row.meta.sender << ',' << row.meta.sequence << ','
           << row.meta.generationTime << ',' << row.meta.size << '\n';
    }

    std::ofstream rx(outputDirectory + "/rx.csv");
    rx << "uid,tx_id,rx_id,seq,t_gen_s,t_rx_s,delay_s,size_bytes\n" << std::setprecision(12);
    for (const auto& row : g_rxRecords)
    {
        rx << row.uid << ',' << row.sender << ',' << row.receiver << ',' << row.sequence << ','
           << row.generationTime << ',' << row.receptionTime << ','
           << (row.receptionTime - row.generationTime) << ',' << row.size << '\n';
    }

    std::ofstream metadata(outputDirectory + "/metadata.json");
    metadata << std::boolalpha << std::setprecision(12)
             << "{\n"
             << "  \"protocol_version\": \"" << protocolVersion << "\",\n"
             << "  \"simulator\": \"ns-3.47\",\n"
             << "  \"regime\": \"" << regime << "\",\n"
             << "  \"seed\": " << seed << ",\n"
             << "  \"probe_nodes\": " << probeNodes << ",\n"
             << "  \"background_nodes\": " << backgroundNodes << ",\n"
             << "  \"background_rate_hz\": " << backgroundRate << ",\n"
             << "  \"background_radius_m\": " << backgroundRadius << ",\n"
             << "  \"background_placement\": \"deterministic_golden_angle_lattice\",\n"
             << "  \"packet_size_bytes\": " << packetSize << ",\n"
             << "  \"probe_rate_hz\": " << probeRate << ",\n"
             << "  \"generation_jitter_fraction\": " << generationJitterFraction << ",\n"
             << "  \"generation_schedule\": \"nonaccumulating_nominal_grid_jitter\",\n"
             << "  \"warmup_s\": " << warmup << ",\n"
             << "  \"measurement_s\": " << measurement << ",\n"
             << "  \"lateral_spacing_m\": " << spacing << ",\n"
             << "  \"standard\": \"IEEE 802.11p\",\n"
             << "  \"channel_number\": 172,\n"
             << "  \"channel_width_mhz\": 10,\n"
             << "  \"center_frequency_mhz\": 5860,\n"
             << "  \"data_mode\": \"OfdmRate6MbpsBW10MHz\",\n"
             << "  \"path_loss_exponent\": " << pathLossExponent << ",\n"
             << "  \"reference_loss_db\": " << referenceLoss << ",\n"
             << "  \"tx_power_dbm\": " << txPower << ",\n"
             << "  \"rx_noise_figure_db\": " << noiseFigure << ",\n"
             << "  \"probe_tx_packets\": " << g_txRecords.size() << ",\n"
             << "  \"probe_rx_events\": " << g_rxRecords.size() << ",\n"
             << "  \"socket_send_failures\": " << g_sendFailures << "\n"
             << "}\n";
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string outputDirectory = "g3b-output";
    std::string protocolVersion = "g3b_calibration_v1.0";
    std::string regime = "light";
    uint64_t seed = 1;
    uint32_t probeNodes = 8;
    uint32_t backgroundNodes = 0;
    double backgroundRate = 0.0;
    double backgroundRadius = 60.0;
    uint32_t packetSize = 200;
    double probeRate = 10.0;
    double warmup = 2.0;
    double measurement = 20.0;
    double spacing = 3.6;
    double pathLossExponent = 2.2;
    double referenceLoss = 46.6777;
    double txPower = 20.0;
    double noiseFigure = 7.0;
    double generationJitterFraction = 0.0;

    CommandLine command(__FILE__);
    command.AddValue("output", "Output directory", outputDirectory);
    command.AddValue("protocolVersion", "Trace protocol version", protocolVersion);
    command.AddValue("regime", "Frozen channel-load regime", regime);
    command.AddValue("seed", "ns-3 run number", seed);
    command.AddValue("probeNodes", "Number of convoy probe vehicles", probeNodes);
    command.AddValue("backgroundNodes", "Number of background transmitters", backgroundNodes);
    command.AddValue("backgroundRate", "Background beacon rate in Hz", backgroundRate);
    command.AddValue("backgroundRadius", "Background-node placement radius in m", backgroundRadius);
    command.AddValue("packetSize", "Beacon size in bytes", packetSize);
    command.AddValue("probeRate", "Probe beacon rate in Hz", probeRate);
    command.AddValue("warmup", "Warm-up duration in s", warmup);
    command.AddValue("measurement", "Measurement duration in s", measurement);
    command.AddValue("spacing", "Probe-vehicle lateral spacing in m", spacing);
    command.AddValue("pathLossExponent", "Log-distance path-loss exponent", pathLossExponent);
    command.AddValue("referenceLoss", "Reference loss in dB", referenceLoss);
    command.AddValue("txPower", "Transmit power in dBm", txPower);
    command.AddValue("noiseFigure", "Receiver noise figure in dB", noiseFigure);
    command.AddValue("generationJitterFraction",
                     "Independent per-beacon jitter as a fraction of the nominal interval",
                     generationJitterFraction);
    command.Parse(argc, argv);

    if (probeNodes < 2 || probeRate <= 0 || measurement <= 0 || packetSize < 16 ||
        (backgroundNodes > 0 && backgroundRate <= 0) || generationJitterFraction < 0.0 ||
        generationJitterFraction >= 0.5)
    {
        std::cerr << "Invalid calibration arguments" << std::endl;
        return 2;
    }

    RngSeedManager::SetSeed(20260807);
    RngSeedManager::SetRun(seed);

    const uint32_t totalNodes = probeNodes + backgroundNodes;
    NodeContainer nodes;
    nodes.Create(totalNodes);

    Ptr<ListPositionAllocator> positions = CreateObject<ListPositionAllocator>();
    const double center = 0.5 * static_cast<double>(probeNodes - 1);
    for (uint32_t i = 0; i < probeNodes; ++i)
    {
        positions->Add(Vector(0.0, (static_cast<double>(i) - center) * spacing, 0.0));
    }
    const double goldenAngle = M_PI * (3.0 - std::sqrt(5.0));
    for (uint32_t i = 0; i < backgroundNodes; ++i)
    {
        const double radius = backgroundRadius *
                              std::sqrt((static_cast<double>(i) + 0.5) /
                                        static_cast<double>(backgroundNodes));
        const double angle = goldenAngle * static_cast<double>(i);
        positions->Add(Vector(radius * std::cos(angle), radius * std::sin(angle), 0.0));
    }

    MobilityHelper mobility;
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    YansWifiChannelHelper channel;
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent",
                               DoubleValue(pathLossExponent),
                               "ReferenceLoss",
                               DoubleValue(referenceLoss));

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("ChannelSettings", StringValue("{172, 10, BAND_5GHZ, 0}"));
    phy.Set("TxPowerStart", DoubleValue(txPower));
    phy.Set("TxPowerEnd", DoubleValue(txPower));
    phy.Set("RxNoiseFigure", DoubleValue(noiseFigure));

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

    InternetStackHelper internet;
    internet.Install(nodes);
    Ipv4AddressHelper addresses;
    addresses.SetBase("10.1.0.0", "255.255.255.0");
    addresses.Assign(devices);
    g_sequences.assign(totalNodes, 0);
    g_probeNodes = probeNodes;

    Ptr<UniformRandomVariable> phase = CreateObject<UniformRandomVariable>();
    phase->SetStream(7003);
    Ptr<UniformRandomVariable> generationJitter = CreateObject<UniformRandomVariable>();
    generationJitter->SetStream(7004);
    const double stopTime = warmup + measurement;
    const uint16_t port = 5000;
    for (uint32_t i = 0; i < totalNodes; ++i)
    {
        Ptr<Socket> sink = Socket::CreateSocket(nodes.Get(i), UdpSocketFactory::GetTypeId());
        sink->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
        sink->SetRecvCallback(MakeBoundCallback(&ReceivePacket, i));

        Ptr<Socket> source = Socket::CreateSocket(nodes.Get(i), UdpSocketFactory::GetTypeId());
        source->SetAllowBroadcast(true);
        source->Connect(InetSocketAddress(Ipv4Address("10.1.0.255"), port));

        const bool probe = i < probeNodes;
        const double rate = probe ? probeRate : backgroundRate;
        const double interval = 1.0 / rate;
        const double jitterAmplitude = generationJitterFraction * interval;
        const double initialPhase = phase->GetValue(jitterAmplitude, interval - jitterAmplitude);
        const uint32_t packetCount = static_cast<uint32_t>(std::llround(measurement * rate));
        for (uint32_t packetIndex = 0; packetIndex < packetCount; ++packetIndex)
        {
            const double perturbation = jitterAmplitude > 0.0
                                            ? generationJitter->GetValue(-jitterAmplitude, jitterAmplitude)
                                            : 0.0;
            const double sendTime = warmup + initialPhase + packetIndex * interval + perturbation;
            Simulator::Schedule(Seconds(sendTime),
                                &SendBeaconOnce,
                                i,
                                source,
                                probe,
                                packetSize);
        }
    }

    Simulator::Stop(Seconds(stopTime + 0.5));
    Simulator::Run();
    Simulator::Destroy();

    WriteOutputs(outputDirectory,
                 protocolVersion,
                 regime,
                 seed,
                 probeNodes,
                 backgroundNodes,
                 backgroundRate,
                 backgroundRadius,
                 packetSize,
                 probeRate,
                 warmup,
                 measurement,
                 spacing,
                 pathLossExponent,
                 referenceLoss,
                 txPower,
                 noiseFigure,
                 generationJitterFraction);

    const uint64_t expected = static_cast<uint64_t>(g_txRecords.size()) * (probeNodes - 1);
    const double pdr = expected > 0 ? static_cast<double>(g_rxRecords.size()) / expected : 0.0;
    std::cout << std::fixed << std::setprecision(6)
              << "G3B_NS3_OK regime=" << regime << " seed=" << seed
              << " tx=" << g_txRecords.size() << " expected=" << expected
              << " received=" << g_rxRecords.size() << " pdr=" << pdr
              << " send_failures=" << g_sendFailures << std::endl;
    return 0;
}
