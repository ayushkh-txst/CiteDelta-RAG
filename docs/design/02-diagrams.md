# 02 — Diagrams

## Level 1 — Context

```mermaid
graph LR
    User[Person asking about<br/>immigration regulations]
    CD[CiteDelta]
    ECFR[eCFR versioner API<br/>ecfr.gov]
    LLM[Hosted LLM]

    User -->|question + as-of date| CD
    CD -->|answer with verified citations,<br/>or a refusal| User
    CD -->|fetch section versions| ECFR
    CD -->|generate from admissible chunks| LLM
```

## Level 2 — Containers

```mermaid
graph TB
    subgraph CiteDelta
        API[FastAPI<br/>HTTP + server-rendered UI]
        W[Queue worker<br/>ingest + parse]
        IDX[(Index files<br/>mmap postings, vectors)]
        PG[(PostgreSQL 17<br/>bitemporal corpus, jobs, traces)]
    end
    U[User] --> API
    API --> IDX
    API --> PG
    W --> PG
    W --> ECFR[eCFR API]
    API --> LLM[Hosted LLM]
```

## Level 3 — The retrieval path

```mermaid
graph LR
    Q[query + as_of] --> ADM[AdmissibleSet<br/>ids in force at as_of]
    Q --> EMB[LocalEmbeddings<br/>ONNX, 384-dim]
    ADM --> CF1[compile_filter → mask]
    ADM --> CF2[compile_filter → mask]
    CF1 --> BM[LexicalIndex<br/>BM25, filter in traversal]
    CF2 --> VEC[VectorIndex<br/>filter in traversal]
    EMB --> VEC
    BM --> RRF[RRF fusion]
    VEC --> RRF
    RRF --> GATE[confidence gate]
    GATE --> GEN[cite-or-refuse]
    GEN --> VAL[citation validator]
    VAL --> OUT[Answer or Refusal]
```

## Sequence — one question

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant P as Postgres
    participant I as Indexes
    participant L as LLM

    U->>A: POST /ask {query, as_of}
    A->>P: admissible ids at as_of
    A->>A: embed query (ONNX, in-process)
    A->>I: BM25 search (mask pushed into traversal)
    A->>I: vector search (mask pushed into traversal)
    A->>A: RRF fuse
    alt no admissible hits, or top score below threshold
        A-->>U: Refusal (no tokens spent)
    else
        A->>P: hydrate chunk text + effective ranges
        A->>L: system prompt + admissible excerpts, ids assigned
        L-->>A: {sufficient, answer, citation_ids}
        A->>A: validate every cited id
        alt a citation fails validation
            A-->>U: Refusal (answer discarded)
        else
            A->>P: persist trace
            A-->>U: Answer + citations
        end
    end
```

The `alt` branches are the point: the happy path alone describes a demo. The
refusal branches are where the design lives — the confidence gate keeps tokens
unspent, and the citation validator guarantees every cited id is one of the
admissible chunks actually shown to the model.
