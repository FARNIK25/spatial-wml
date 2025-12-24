# Subject Evaluation Diagrams

## Pipeline Overview

```mermaid
flowchart LR
    A[Checkpoint (.pth)] --> B[Model Load]
    B --> C[Slice Loader (val/test split)]
    C --> D[Slice Inference]
    D --> E[Per-slice Log]
    E --> F{Group by Subject}
    F --> G[Vote Counts]
    F --> H[Probability Sums]
    G & H --> I[Majority Vote + Tie Break]
    I --> J[Subject Prediction]
    J --> K[Confusion + Accuracy]
    J --> L[Subject CSV]
    E --> M[Slice CSV]
    K --> N[Metrics JSON]
```

## Slice Log Schema

```mermaid
classDiagram
    class SliceRecord {
        string subject_id
        int slice_index
        string file_name
        string file_path
        string true_label
        string pred_label
        float pred_confidence
        bool correct
        map prob_class -> float
    }

    class SubjectVote {
        string subject_id
        int num_slices
        map votes_class -> int
        map mean_prob_class -> float
        string true_label
        string pred_label
        bool correct
    }

    SliceRecord "*" --> "1" SubjectVote : aggregates
```
