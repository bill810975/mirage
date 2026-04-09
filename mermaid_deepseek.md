flowchart TB

  %% =========================
  %% Input / Embedding / Final Head
  %% =========================
  subgraph IO["Input / Output"]
    EWT["input_layernorm.weight<br/>[129280, 7168] bf16"]
    EMB["Embedding"]
    H0["hidden states<br/>[N, 7168] bf16"]

    NORMF["RMSNorm"]
    NORMFW["norm.weight<br/>[7168] bf16"]
    HF["Output<br/>[N, 7168] bf16"]

    LMH["Linear"]
    LMHW["lm_head.weight<br/>[129280, 7168] bf16"]
    LOGITS["Logits<br/>[N, 129280] bf16"]

    EWT --> EMB --> H0
    H0 --> NORMF --> HF --> LMH --> LOGITS
    NORMFW --> NORMF
    LMHW --> LMH
  end

  %% =========================
  %% 61 transformer layers
  %% =========================
  HF --> LSTACK["x61 layers in total"]

  %% =========================
  %% One transformer layer
  %% =========================
  subgraph LAYER["One Transformer Layer"]
    direction TB

    %% -------------------------
    %% Pre-attention norm
    %% -------------------------
    HIN["hidden states<br/>[N, 7168] bf16"]
    INN["RMSNorm"]
    INNW["input_layernorm.weight<br/>[7168] bf16"]
    HN["hidden states<br/>[N, 7168] bf16"]

    HIN --> INN --> HN
    INNW --> INN

    %% -------------------------
    %% MLA / Attention
    %% -------------------------
    subgraph MLA["MLA Attention (Absorb + Distributed + Quant)"]
      direction TB

      %% q_a / kv_a fused projection
      FQKV["Fused Linear"]
      FWQ["q_a_proj.weight<br/>[1536, 7168] fp8e4m3"]
      FSKQ["q_a_proj.scale<br/>[12, 56] fp32"]
      FWKV["kv_a_proj_with_mqa.weight<br/>[576, 7168] fp8e4m3"]
      FSKV["kv_a_proj_with_mqa.scale<br/>[5, 56] fp32"]

      CQ["c_Q<br/>[N, 1536] bf16"]
      CKV0["c_KV<br/>[N, 512] bf16"]
      KR["k_R<br/>[N, 64] bf16"]

      HN --> FQKV
      FWQ --> FQKV
      FSKQ --> FQKV
      FWKV --> FQKV
      FSKV --> FQKV

      FQKV --> CQ
      FQKV --> CKV0
      FQKV --> KR

      %% q_a path
      QAN["RMSNorm"]
      QANW["q_a_layernorm.weight<br/>[1536] bf16"]
      QB["Linear"]
      QBW["q_b_proj.weight<br/>[128//TP*192, 1536] fp8e4m3"]
      QBS["q_b_proj.scale<br/>[192//TP, 12] fp32"]

      QFULL["q<br/>[N, n_h=128//TP, (128+64)] bf16"]
      QNOPE0["q_nope<br/>[128//TP, N, 128] bf16"]
      QPE0["q_pe<br/>[128//TP, N, 64] bf16"]

      CQ --> QAN --> QB --> QFULL
      QANW --> QAN
      QBW --> QB
      QBS --> QB

      QFULL --> QNOPE0
      QFULL --> QPE0

      %% kv_a path
      KVAN["RMSNorm"]
      KVANW["kv_a_layernorm.weight<br/>[512] bf16"]
      CKV["c_KV<br/>[N, 512] bf16"]

      CKV0 --> KVAN --> CKV
      KVANW --> KVAN

      %% rope + cache
      ROPE["RoPE"]
      KPE_N["k_pe<br/>[N, 64] bf16"]

      KR --> ROPE --> KPE_N
      QPE0 --> ROPE

      CACHE["Cache<br/>bf16"]
      CKV --> CACHE
      KPE_N --> CACHE

      CKV_S["c_KV<br/>[S, 512] bf16"]
      KPE_S["k_pe<br/>[S, 64] bf16"]

      CACHE --> CKV_S
      CACHE --> KPE_S

      %% absorbed / broadcasted K,V
      KNOPE["k_nope = c_KV<br/>[bcst=128//TP, S, 512] bf16"]
      KPE_B["k_pe<br/>[bcst=128, S, 64] bf16"]
      VBC["V = c_KV<br/>[bcst=128//TP, S, 512] bf16"]

      CKV_S --> KNOPE
      CKV_S --> VBC
      KPE_S --> KPE_B

      %% q_nope absorb path
      QNOPE_IN["q_nope<br/>[128//TP, N, 128] bf16"]
      QNOPE_L["Linear"]
      QNOPE_W["kv_b_proj.weight.split(2)[0].T<br/>[128//TP*128, 512]<br/>([128//TP, 512, 128]) fp8e4m3"]
      QNOPE_S["kv_b_proj.scale.split(2)[0].T<br/>[128//TP, 4] fp32"]
      QNOPE_512["q_nope<br/>[128//TP, N, 512] bf16"]

      QNOPE0 --> QNOPE_IN --> QNOPE_L --> QNOPE_512
      QNOPE_W --> QNOPE_L
      QNOPE_S --> QNOPE_L

      %% attention core
      MQA["MQA"]
      QCAT["Q<br/>[128//TP, N, 576]"]
      KCAT["K<br/>[128//TP, S, 576]"]
      VCAT["V<br/>[128//TP, S, 512]"]
      ATTN0["attn_out<br/>[N, 128//TP, 512] bf16"]

      QNOPE_512 --> MQA
      QPE0 --> MQA
      KNOPE --> MQA
      KPE_B --> MQA
      VBC --> MQA

      MQA --> ATTN0

      %% output projection inside attention
      AO_L["Linear"]
      AO_W["kv_b_proj.weight.split(2)[1]<br/>[128//TP*128, 512]<br/>([128//TP,128,512]) fp8e4m3"]
      AO_S["kv_b_proj.scale.split(2)[1]<br/>[128//TP, 4] fp32"]
      ATTN1["attn_out<br/>[N, 128//TP*128] bf16"]

      O_L["Linear"]
      O_W["o_proj.weight<br/>[7168, 128//TP*128] fp8e4m3"]
      O_S["kv_b_proj.scale<br/>[56, 128//TP] fp32"]
      O_PART["attn_out<br/>[N, 128*128] bf16"]
      O_AR["Allreduce"]
      O["o<br/>[N, 7168] bf16"]

      ATTN0 --> AO_L --> ATTN1 --> O_L --> O_PART --> O_AR --> O
      AO_W --> AO_L
      AO_S --> AO_L
      O_W --> O_L
      O_S --> O_L

      %% residual
      RES["Residual"]
      HN --> RES
      O --> RES
      RES --> ORES["o<br/>[N, 7168] bf16"]

    end

    %% -------------------------
    %% Post-attention norm
    %% -------------------------
    PAN["RMSNorm"]
    PANW["post_attention_layernorm.weight<br/>[7168] bf16"]
    OPOST["o<br/>[N, 7168] bf16"]

    ORES --> PAN --> OPOST
    PANW --> PAN

    %% -------------------------
    %% Layer-type split
    %% -------------------------
    DENSE_NOTE["The first 3 layers are dense MLP"]
    MOE_NOTE["The next 58 layers are MoE"]

    OPOST --> DENSE_NOTE
    OPOST --> MOE_NOTE

    %% -------------------------
    %% Dense MLP path
    %% -------------------------
    subgraph DENSE["Dense MLP"]
      direction TB

      Q1["quantize"]
      HFP8["hidden states<br/>[N, 7168] fp8_e4m3fn"]
      HSCALE["scales<br/>[N, 7168//blocksize] ue8m0"]

      W13["Fused Linear"]
      GW["gate_proj.weight<br/>[18432//TP, 7168] fp8e4m3"]
      GS["gate_proj.scale<br/>[144//TP, 56] fp32"]
      UW["up_proj.weight<br/>[18432//TP, 7168] fp8e4m3"]
      US["up_proj.scale<br/>[144//TP, 56] fp32"]

      GATE["gate<br/>[N, 18432//TP] bf16"]
      UP["up<br/>[N, 18432//TP] bf16"]

      ACT["Fused silu_mul"]
      HMID["hidden state<br/>[N, 18432//TP] bf16"]

      Q2["quantize"]
      HMID_FP8["hidden states<br/>[N, 18432//TP] fp8_e4m3fn"]
      HMID_SCALE["scales<br/>[N, 18432//TP//blocksize] ue8m0"]

      W2["Linear"]
      DW["down_proj.weight<br/>[7168, 18432//TP] fp8e4m3"]
      DOUT["Output<br/>[N, 7168] bf16"]
      DAR["Allreduce"]

      OPOST --> Q1 --> HFP8
      Q1 --> HSCALE

      HFP8 --> W13
      HSCALE --> W13
      GW --> W13
      GS --> W13
      UW --> W13
      US --> W13

      W13 --> GATE
      W13 --> UP
      GATE --> ACT
      UP --> ACT
      ACT --> HMID

      HMID --> Q2 --> HMID_FP8
      Q2 --> HMID_SCALE

      HMID_FP8 --> W2 --> DOUT --> DAR
      HMID_SCALE --> W2
      DW --> W2
    end

    %% -------------------------
    %% MoE path
    %% -------------------------
    subgraph MOE["MoE + Shared Expert"]
      direction TB

      GATEL["Gate Linear"]
      GATEW["gate.weight<br/>[256, 7168] bf16"]
      GOUT["gate_out(num_exp=256)<br/>[N, 256] bf16"]

      TOPK["topk & sigmoid"]
      BIAS["gate.e_score_correction_bias<br/>[256] FP32"]
      TKW["topk_weights (k=8)<br/>[N, 8] bf16"]
      RIDX["moe_routing_idx<br/>[256, N] bf16"]
      MMASK["moe_mask<br/>[256+1] bf16"]

      NOTE_SIG["deepseek v3 uses Sigmoid instead of Softmax"]
      NOTE_TOPK["Use only for topk routing, not topk weights"]

      OPOST --> GATEL --> GOUT --> TOPK
      GATEW --> GATEL
      BIAS --> TOPK
      TOPK --> TKW
      TOPK --> RIDX
      TOPK --> MMASK
      NOTE_SIG --> TOPK
      NOTE_TOPK --> TOPK

      %% local experts
      EXP_NOTE["experts x256<br/>num_exp_per_token=8"]
      W13E["moe_w13_linear"]
      EWG["gate_proj.weight<br/>[256//EPmoe, 2048//TPmoe, 7168] fp8e4m3"]
      EWG_S["gate_proj.scale<br/>[256//EPmoe, 16//TPmoe, 56] fp32"]
      EWU["up_proj.weight<br/>[256//EPmoe, 2048//TPmoe, 7168] fp8e4m3"]
      EWU_S["up_proj.scale<br/>[256//EPmoe, 16//TPmoe, 56] fp32"]

      EGATE["gate<br/>[N, 8, 2048//TPmoe] bf16"]
      EUP["up<br/>[N, 8, 2048//TPmoe] bf16"]
      EACT["Fused silu_mul"]
      EMID["silu_mul_out<br/>[N, 8, 2048//TPmoe] bf16"]

      QE["quantize"]
      EMID_FP8["hidden states<br/>[N, 8, 2048//TPmoe] fp8_e4m3fn"]
      EMID_SC["scales<br/>[N, 8, 2048//TPmoe//blocksize] ue8m0"]

      W2E["moe_w2_linear"]
      EOUT["mlp_out<br/>[N, 8, 7168] bf16"]

      RIDX --> W13E
      MMASK --> W13E
      EWG --> W13E
      EWG_S --> W13E
      EWU --> W13E
      EWU_S --> W13E

      OPOST --> W13E
      W13E --> EGATE
      W13E --> EUP
      EGATE --> EACT
      EUP --> EACT
      EACT --> EMID --> QE --> EMID_FP8
      QE --> EMID_SC

      EMID_FP8 --> W2E --> EOUT
      EMID_SC --> W2E

      EXP_NOTE --> W13E

      %% shared expert
      SH_NOTE["shared_expert x1"]
      SH_COMM["Shared Expert are TP across all GPUs,<br/>while experts are groupwise TP"]

      W13S["Fused Linear"]
      SWG["gate_proj.weight<br/>[2048//TP, 7168] fp8e4m3"]
      SWG_S["gate_proj.scale<br/>[16//TP, 56] fp32"]
      SWU["up_proj.weight<br/>[2048//TP, 7168] fp8e4m3"]
      SWU_S["up_proj.scale<br/>[16//TP, 56] fp32"]

      SGATE["gate<br/>[N, 2048//TP] bf16"]
      SUP["up<br/>[N, 2048//TP] bf16"]
      SACT["Fused silu_mul"]
      SMID["silu_mul_out<br/>[N, 2048//TP] bf16"]

      QS["quantize"]
      SMID_FP8["hidden states<br/>[N, 2048//TP] fp8_e4m3fn"]
      SMID_SC["scales<br/>[N, 2048//TP] ue8m0"]

      W2S["moe_w13_linear"]
      SOUT["mlp_out<br/>[N, 1, 7168] bf16"]

      OPOST --> W13S
      SWG --> W13S
      SWG_S --> W13S
      SWU --> W13S
      SWU_S --> W13S

      W13S --> SGATE
      W13S --> SUP
      SGATE --> SACT
      SUP --> SACT
      SACT --> SMID --> QS --> SMID_FP8
      QS --> SMID_SC

      SMID_FP8 --> W2S --> SOUT
      SMID_SC --> W2S

      SH_NOTE --> W13S
      SH_COMM --> W13S

      %% reduction
      MRED["moe_local_reduction"]
      MOUT["Output<br/>[N, 7168] bf16"]
      MAR["Allreduce"]

      TKW --> MRED
      EOUT --> MRED
      SOUT --> MRED
      OPOST --> MRED

      MRED --> MOUT --> MAR
    end

    %% layer output
    DAR --> LOUT["Output<br/>[N, 7168] bf16"]
    MAR --> LOUT

  end

  %% =========================
  %% Optional MTP
  %% =========================
  subgraph MTP["Optional MTP"]
    direction TB
    MTPNOTE["Optional MTP layer with EAGLE spec decode"]
  end

  MTPNOTE --> INN
