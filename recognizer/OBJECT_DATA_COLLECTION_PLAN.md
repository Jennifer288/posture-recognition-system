# Object Data Collection Plan

Purpose: validate and tune `OccupancyDetector` HUMAN vs OBJECT rules without changing the frozen posture dataset.

Store these CSV files in a separate object-data directory, not inside `dataset_v1_1_17_final`.

## Hardware Detection Boundary

The cushion has a minimum detectable load. If a physical object produces no visible numeric response or stable pressure region in the 16x16 matrix, software cannot observe it.

For the software:

- No sensor response is equivalent to `EMPTY`.
- A weak but repeatable nonzero response can be `LOAD_BELOW_THRESHOLD`.
- Only a stable detectable pressure map enters `OBJECT` / `UNKNOWN` validation.
- Do not record CSVs for objects that create no stable nonzero pressure region.

## Object Categories

### A. Below Hardware Detection Threshold

Examples may include very light clothes, light paper, or small light objects. If the matrix does not show a stable nonzero pressure region, do not record these as object CSVs. This is a hardware sensing limit, not an occupancy software failure.

### B. Detectable Non-Human Loads

Each detectable object should have 2 independent CSV files first:

- Loaded backpack: `object_backpack_loaded_01.csv`
- Large stack of books: `object_books_heavy_01.csv`
- Loaded cardboard box or hard case: `object_box_loaded_01.csv`
- Rice bag / water bucket / dumbbell: `object_heavy_single_01.csv`

Before recording, manually confirm that placing the object creates a stable nonzero 16x16 pressure map.

## Recording Flow

```text
empty cushion 2s
-> place object naturally
-> keep object still 10s
-> remove object
-> empty cushion 2s
```

## Notes

- Keep object data separate from human posture labels.
- Do not train or tune posture RF with object data.
- Use object data only for EMPTY / LOAD_BELOW_THRESHOLD / OBJECT / HUMAN / UNKNOWN gate validation.
- Record object orientation and approximate placement in a small side note if possible.
