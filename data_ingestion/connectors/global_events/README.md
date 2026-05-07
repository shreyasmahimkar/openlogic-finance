# Global Events Agent
> "One plot is worth a 1000 words."

## Architecture Rationale
In multi-agent systems, data ingestion and data visualization often become tightly coupled. However, the fundamental role of ingesting high-throughput numerical market streams requires low latency and strict isolation, whereas generating rich, contextual visualizations demands a deep, qualitative reasoning capability. 

The `global_events_agent` embraces **Separation of Concerns**. By decoupling the plotting logic out of the `market_data` agent, we allow the visualization process to evaluate a broader temporal and semantic context. 

## Features
- **Global Context Infusion**: Natively reads `global_events.csv` to overlay socio-economic regimes onto raw candlestick lines.
- **Dynamic Search Gap-Filling**: Built-in conditional triggers to fetch live data beyond its static temporal boundary (May 2026+), ensuring that visual charts are not just data-dumps, but rich, context-aware infographics that visually map the "Why" behind the "What."
