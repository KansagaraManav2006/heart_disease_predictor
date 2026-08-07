# ADR-001: HealthLens AI Architecture & Technology Stack

**Date:** 2026-08-07  
**Status:** Accepted  
**Deciders:** Core Engineering Team  

---

## Context

The legacy project was built as a basic prototype featuring an Express backend spawning child Python processes per request, flat-file JSON persistence, unauthenticated routes, and hardcoded ML preprocessing.

To transition the platform into **HealthLens AI** — a research-grade cardiometabolic risk-intelligence and clinical decision-support platform — a scalable, secure, and maintainable architecture is required.

---

## Decision

We adopt a decoupled, research-pilot architecture consisting of:

1. **Frontend:** React 19.2 + Vite 7 + Tailwind CSS v4. Managed with TanStack Query for server-state caching and Zod / React Hook Form for schema-driven validation.
2. **Public API Gateway & Application Server:** Node.js (v24 LTS) + Express + TypeScript. Handles authentication, server-side session management (PostgreSQL-backed), RBAC, input validation, CSRF protection, and audit logging.
3. **Database & ORM:** PostgreSQL 17 managed via Prisma ORM. Ensures ACID compliance, relational integrity (users, sessions, profiles, grants, assessments, model versions, audit events), and `pgvector` support for future RAG features.
4. **Internal ML/OCR Service:** Python FastAPI service operating behind the API gateway (inaccessible from the public internet). Loads trained scikit-learn model pipelines once into memory at startup, exposes Pydantic validation schemas, and computes SHAP feature importance.
5. **Security & Privacy:**
   - Password hashing: Argon2id.
   - Authentication: Server-side HTTP-only cookies with session rotation.
   - Authorization: Attribute & Role-Based Access Control (RBAC) supporting Patient, Clinician, and Admin personas.
   - CSRF: Synchronizer-token pattern on mutating requests.
   - History: Isolated per user/patient; no committed mock patient data; session/DB history strictly scoped.

---

## Consequences

### Positive
- Prevents resource-exhaustion overhead from spawning Python subprocesses per HTTP request.
- Provides strict tenant and patient data isolation.
- Establishes a versioned model registry and reproducible pipeline lineage.
- Complies with OWASP ASVS Level 2 security guidelines.

### Negative / Trade-offs
- Requires running PostgreSQL (via Docker Compose or local instance).
- Adds typed schema maintenance between TypeScript (Zod/Prisma) and Python (Pydantic).
- Model artifact updates require explicit registry versioning and service reload.

---

## Compliance & Scope Disclaimer

HealthLens AI is built as a **research and portfolio pilot**. It is NOT a FDA-cleared medical device, nor does it provide autonomous clinical diagnosis or treatment prescribing.
