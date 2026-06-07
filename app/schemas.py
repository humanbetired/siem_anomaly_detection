from pydantic import BaseModel, Field
from typing import Optional


class NetworkFlow(BaseModel):
    """Input: raw network flow features."""
    bwd_packet_length_mean: float = Field(alias="Bwd Packet Length Mean")
    avg_bwd_segment_size: float = Field(alias="Avg Bwd Segment Size")
    bwd_packet_length_max: float = Field(alias="Bwd Packet Length Max")
    bwd_packet_length_std: float = Field(alias="Bwd Packet Length Std")
    destination_port: float = Field(alias="Destination Port")
    urg_flag_count: float = Field(alias="URG Flag Count")
    packet_length_mean: float = Field(alias="Packet Length Mean")
    average_packet_size: float = Field(alias="Average Packet Size")
    packet_length_std: float = Field(alias="Packet Length Std")
    min_packet_length: float = Field(alias="Min Packet Length")
    max_packet_length: float = Field(alias="Max Packet Length")
    packet_length_variance: float = Field(alias="Packet Length Variance")
    min_seg_size_forward: float = Field(alias="min_seg_size_forward")
    bwd_packet_length_min: float = Field(alias="Bwd Packet Length Min")
    avg_fwd_segment_size: float = Field(alias="Avg Fwd Segment Size")
    fwd_packet_length_mean: float = Field(alias="Fwd Packet Length Mean")
    bwd_packets_per_s: float = Field(alias="Bwd Packets/s")
    flow_packets_per_s: float = Field(alias="Flow Packets/s")
    flow_iat_std: float = Field(alias="Flow IAT Std")
    fwd_iat_total: float = Field(alias="Fwd IAT Total")

    class Config:
        populate_by_name = True


class AnomalyResponse(BaseModel):
    """Output: hasil deteksi anomali."""
    is_anomaly: bool
    severity: str                    # LOW, MEDIUM, HIGH, CRITICAL
    anomaly_score: float             # 0.0 - 1.0
    isolation_forest_result: str     # NORMAL / ANOMALY
    autoencoder_result: str          # NORMAL / ANOMALY
    reconstruction_error: float
    message: str


class StatsResponse(BaseModel):
    total_requests: int
    total_anomalies: int
    anomaly_rate: float
    last_alert: Optional[str]