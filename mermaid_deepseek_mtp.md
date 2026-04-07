flowchart TB

  %% =========================
  %% Main Decode Step
  %% =========================
  subgraph MAIN["Main Model Decode (61 layers)"]
    direction TB

    TOK_IN["input token<br/>[1] int64"]
    EMB["Embedding<br/>(shared with MTP)"]
    LAYERS["61 Transformer Layers<br/>(MLA + MoE/Dense MLP)"]
    NORM_F["RMSNorm<br/>model.norm"]
    LM_HEAD["Linear (lm_head)<br/>(shared with MTP)"]
    ARGMAX0["Argmax"]

    HIDDEN["hidden_states<br/>[1, 7168] bf16"]
    TOKEN0["main_token<br/>[1] int64"]

    TOK_IN --> EMB --> LAYERS --> HIDDEN
    HIDDEN --> NORM_F --> LM_HEAD --> ARGMAX0 --> TOKEN0
  end

  %% =========================
  %% MTP Draft Generation (Sequential)
  %% =========================
  subgraph MTP_DRAFT["MTP Draft (K steps, sequential)"]
    direction TB

    NOTE_SEQ["Each step auto-regressive:<br/>output of step i feeds step i+1"]

    %% --- Draft Step 0 ---
    subgraph STEP0["Draft Step 0"]
      direction TB

      %% Inputs
      DT0["draft_input = main_token"]
      HS0["prev_hidden = HIDDEN<br/>(from main model)"]

      %% enorm path
      EMB0["Embed(draft_input)<br/>(shared embed_tokens)"]
      ENORM0["RMSNorm<br/>enorm.weight"]
      E0["enorm_out<br/>[1, 7168]"]

      %% hnorm path
      HNORM0["RMSNorm<br/>hnorm.weight"]
      H0["hnorm_out<br/>[1, 7168]"]

      %% eh_proj: W @ [enorm_out; hnorm_out]
      EH0["Linear eh_proj<br/>[7168, 14336] → [1, 7168]"]

      %% Full decoder layer (MTP block)
      DEC0["DecoderLayer<br/>(MTP layer 0)<br/>MLA attn + MLP"]
      MTP_HS0["mtp_hidden_0<br/>[1, 7168]"]

      %% lm_head → argmax
      NORM0["RMSNorm"]
      LMH0["Linear (shared lm_head)"]
      ARG0["Argmax"]
      DRAFT0["draft_token_0<br/>[1] int64"]

      DT0 --> EMB0 --> ENORM0 --> E0
      HS0 --> HNORM0 --> H0
      E0 --> EH0
      H0 --> EH0
      EH0 --> DEC0 --> MTP_HS0
      MTP_HS0 --> NORM0 --> LMH0 --> ARG0 --> DRAFT0
    end

    %% --- Draft Step 1 ---
    subgraph STEP1["Draft Step 1"]
      direction TB

      DT1["draft_input = draft_token_0"]
      HS1["prev_hidden = mtp_hidden_0"]

      EMB1["Embed(draft_input)"]
      ENORM1["RMSNorm enorm"]
      HNORM1["RMSNorm hnorm"]
      EH1["Linear eh_proj"]
      DEC1["DecoderLayer<br/>(MTP layer 0, recycled)"]
      MTP_HS1["mtp_hidden_1"]
      NORM1["RMSNorm"]
      LMH1["Linear (shared lm_head)"]
      ARG1["Argmax"]
      DRAFT1["draft_token_1"]

      DT1 --> EMB1 --> ENORM1
      HS1 --> HNORM1
      ENORM1 --> EH1
      HNORM1 --> EH1
      EH1 --> DEC1 --> MTP_HS1 --> NORM1 --> LMH1 --> ARG1 --> DRAFT1
    end

    %% Connect steps
    DRAFT0 --> DT1
    MTP_HS0 --> HS1

    STEP1 --> STEPK["... repeat for K draft steps"]
  end

  %% Connect main → MTP
  TOKEN0 --> DT0
  HIDDEN --> HS0

  %% =========================
  %% Target Verification
  %% =========================
  subgraph VERIFY["Target Model Verification"]
    direction TB

    NOTE_VERIFY["Run main model on all draft tokens at once<br/>(multi-token prefill)"]

    DRAFT_IDS["draft_token_ids<br/>[K] int64"]
    MAIN_VERIFY["Main Model Forward<br/>(61 layers, K+1 tokens)"]
    TARGET_IDS["target_token_ids<br/>[K+1] int64<br/>(argmax at each position)"]

    DRAFT_IDS --> MAIN_VERIFY --> TARGET_IDS
  end

  DRAFT0 --> DRAFT_IDS
  DRAFT1 --> DRAFT_IDS

  %% =========================
  %% Accept / Reject
  %% =========================
  subgraph ACCEPT["Verification + Accept/Commit"]
    direction TB

    VERIFY_K["target_verify_strict<br/>or probabilistic<br/>or synthetic"]

    ACC_COUNT["accepted_count<br/>[1] int32"]
    OUT_TOKS["output_tokens<br/>[K+1] int64"]

    COMMIT["mtp_accept_commit"]
    NEW_POS["new_position"]
    FINAL_OUT["final_output_tokens"]
    NUM_NEW["num_new_tokens"]

    DRAFT_IDS2["draft_token_ids"] --> VERIFY_K
    TARGET_IDS2["target_token_ids"] --> VERIFY_K

    VERIFY_K --> ACC_COUNT
    VERIFY_K --> OUT_TOKS

    ACC_COUNT --> COMMIT
    OUT_TOKS --> COMMIT
    COMMIT --> NEW_POS
    COMMIT --> FINAL_OUT
    COMMIT --> NUM_NEW
  end

  TARGET_IDS --> TARGET_IDS2
  DRAFT_IDS --> DRAFT_IDS2

  %% =========================
  %% Loop back
  %% =========================
  FINAL_OUT --> NEXT["Next decode iteration<br/>(feed accepted tokens back)"]
  NEXT --> TOK_IN

  %% =========================
  %% Notes
  %% =========================
  subgraph NOTES["Key Design Notes"]
    direction TB
    N1["1. MTP is sequential: step i depends on step i-1"]
    N2["2. Shared weights: embed_tokens + lm_head"]
    N3["3. MTP-specific weights: enorm, hnorm, eh_proj, decoder_layer"]
    N4["4. DeepSeek V3: num_nextn_predict_layers = 1<br/>layers recycle via modulo"]
    N5["5. KV cache shared with main model"]
    N6["6. In MPK: draft steps statically unrolled at compile time"]
  end
