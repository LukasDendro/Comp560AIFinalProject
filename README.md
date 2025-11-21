# Extracting Root Problems from User Pain Points

An automated pipeline for identifying root causes of user dissatisfaction and generating actionable recommendations from product feedback using Retrieval-Augmented Synthesis.

## Overview

Innovation and product development depend on understanding user needs, but manually analyzing feedback from reviews, support tickets, and surveys is labor-intensive and difficult to scale. This project automates the extraction of root problems from large volumes of user complaints.

Our system uses a **Retrieval-Augmented Synthesis (RAS)** framework: given a query about a specific issue, it employs dense embeddings to retrieve semantically similar historical pain points from a curated database. A Large Language Model then synthesizes these examples using few-shot prompting to identify candidate root causes with evidence-backed suggestions.

## Key Contributions

1. **Curated Pain Point Database**: High-quality, standardized database extracted from large-scale Amazon headphone reviews, rigorously cleaned for systematic analysis.

2. **Context-Aware Synthesis**: RAS framework synthesizes insights across similar historical complaints, yielding more accurate root-cause identification than direct LLM inference on isolated feedback.

3. **Efficient Retrieval**: K-means clustering pre-classifies pain points in embedding space, reducing search scope and improving scalability (~5x speedup).

## Results

In qualitative evaluation on Amazon headphone reviews, the system reliably retrieves coherent complaint neighborhoods and surfaces cross-cutting root problems around reliability, comfort, and communication quality, transforming specific complaints into actionable problem statements for innovation work.


COMP 560 - AI Final Project
Jake Terrill, Lukas Dendrolivanos, Benjamin Corter, Zhen Xu
