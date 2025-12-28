# mlflow_governance.py
Agents governance
"""
TIS-IntelliMat ESG Navigator - MLflow AI Governance
====================================================
File: mlflow_governance.py
Location: Add to /backend/ or /railway-api/ folder

MLflow URL: https://refreshing-laughter-production-fb60.up.railway.app
Principle: "No Data Without Lineage"

Usage:
    from mlflow_governance import track, quick_log, setup_experiments
    
    # Full tracking
    with track("ESG_COMPLIANCE", "Ivanhoe Mines Ltd") as t:
        t.log_score(78).log_confidence(0.87)
    
    # Quick one-liner
    quick_log("ESG_COMPLIANCE", "Sasol Limited", 58, "Chemicals")
"""

import mlflow
from mlflow.tracking import MlflowClient
import json
import uuid
import hashlib
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional

# =============================================================================
# CONFIGURATION - LIVE ENDPOINTS
# =============================================================================

MLFLOW_URI = "https://refreshing-laughter-production-fb60.up.railway.app"
ESG_API = "https://esg-navigator-production.up.railway.app"
APP_URL = "https://app.esgnavigator.ai"

# 8 ESG Navigator AI Agents
AGENTS = {
    "ESG_COMPLIANCE": {"name": "ESG Compliance Agent", "standards": ["GRI", "SASB", "TCFD", "CDP", "JSE"]},
    "SUPPLIER_RISK": {"name": "Supplier Risk Assessment Agent", "standards": ["ISO 20400", "SA8000"]},
    "PARTNERSHIP_EVAL": {"name": "Partnership Evaluation Agent", "standards": ["TIS Framework"]},
    "DATA_ENRICHMENT": {"name": "Data Enrichment Agent", "standards": ["Data Quality"]},
    "BUSINESS_INTELLIGENCE": {"name": "Business Intelligence Agent", "standards": ["BI Best Practices"]},
    "ISO_STANDARDS": {"name": "ISO Standards Agent", "standards": ["ISO 14001", "ISO 45001", "ISO 50001", "ISO 27001"]},
    "GISTM_TAILINGS": {"name": "GISTM Tailings Agent", "standards": ["GISTM", "ICMM"]},
    "ISSB_CLIMATE": {"name": "ISSB Climate Disclosure Agent", "standards": ["IFRS S1", "IFRS S2", "ISSB"]}
}


# =============================================================================
# TRACKER CLASS
# =============================================================================

class ESGTracker:
    """
    MLflow tracker for ESG Navigator AI agents with full governance.
    
    Example:
        tracker = ESGTracker()
        with tracker.start("ESG_COMPLIANCE", "Ivanhoe Mines Ltd") as t:
            t.log_score(78, {"E": 75, "S": 80, "G": 79})
            t.log_confidence(0.87)
            t.log_gaps([{"area": "Water", "severity": "medium"}])
    """
    
    def __init__(self, tracking_uri: str = MLFLOW_URI):
        mlflow.set_tracking_uri(tracking_uri)
        self.session = str(uuid.uuid4())[:8]
        self._run = None
        self._start = None
        
    def start(self, agent: str, company: str, user: str = "system"):
        """Start tracking an analysis"""
        self._agent = agent
        self._company = company
        self._user = user
        return self
    
    def __enter__(self):
        self._start = datetime.utcnow()
        
        # Set experiment
        mlflow.set_experiment(f"TIS_ESG_Navigator/{self._agent}")
        
        # Start run
        self._run = mlflow.start_run(
            run_name=f"{self._company.replace(' ', '_')}_{self._start.strftime('%H%M%S')}"
        )
        
        # Context tags
        mlflow.set_tag("company", self._company)
        mlflow.set_tag("agent", self._agent)
        mlflow.set_tag("agent_name", AGENTS.get(self._agent, {}).get("name", self._agent))
        mlflow.set_tag("user", self._user)
        mlflow.set_tag("session", self.session)
        mlflow.set_tag("timestamp", self._start.isoformat())
        mlflow.set_tag("platform", "TIS-IntelliMat ESG Navigator")
        mlflow.set_tag("principle", "No Data Without Lineage")
        mlflow.log_param("model", "claude-sonnet-4-20250514")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self._start).total_seconds()
        mlflow.log_metric("duration_sec", round(duration, 2))
        
        if exc_type:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val)[:200])
        else:
            mlflow.set_tag("status", "SUCCESS")
        
        mlflow.end_run()
        return False
    
    # -------------------------------------------------------------------------
    # LOGGING METHODS (chainable)
    # -------------------------------------------------------------------------
    
    def log_score(self, score: int, breakdown: Dict[str, int] = None):
        """Log ESG score (0-100) with optional E/S/G breakdown"""
        mlflow.log_metric("esg_score", score)
        
        # Archetype
        arch = "SOPHISTICATED" if score >= 75 else "OPERATIONAL" if score >= 50 else "TRUE_LAGGARD"
        mlflow.set_tag("archetype", arch)
        
        if breakdown:
            for k, v in breakdown.items():
                mlflow.log_metric(f"score_{k.lower()}", v)
        return self
    
    def log_confidence(self, score: float):
        """Log AI confidence (0.0 - 1.0)"""
        mlflow.log_metric("confidence", round(score, 3))
        if score < 0.7:
            mlflow.set_tag("requires_review", "true")
        return self
    
    def log_sector(self, sector: str):
        """Log company sector"""
        mlflow.set_tag("sector", sector)
        return self
    
    def log_standards(self, standards: List[str]):
        """Log compliance standards checked"""
        mlflow.log_param("standards", ", ".join(standards))
        mlflow.log_metric("standards_count", len(standards))
        return self
    
    def log_gaps(self, gaps: List[Dict]):
        """Log compliance gaps found"""
        mlflow.log_metric("gaps_count", len(gaps))
        if gaps:
            high = sum(1 for g in gaps if g.get("severity") == "high")
            mlflow.log_metric("gaps_high", high)
            self._artifact(gaps, "gaps.json", "compliance")
        return self
    
    def log_recommendations(self, recs: List[Dict]):
        """Log AI recommendations"""
        mlflow.log_metric("recs_count", len(recs))
        if recs:
            high = sum(1 for r in recs if r.get("priority") == "high")
            mlflow.log_metric("recs_high", high)
            self._artifact(recs, "recommendations.json", "output")
        return self
    
    def log_tokens(self, input_t: int, output_t: int):
        """Log Anthropic API token usage"""
        mlflow.log_metric("tokens_in", input_t)
        mlflow.log_metric("tokens_out", output_t)
        mlflow.log_metric("tokens_total", input_t + output_t)
        cost = (input_t * 0.003 + output_t * 0.015) / 1000
        mlflow.log_metric("cost_usd", round(cost, 4))
        return self
    
    def log_input(self, data: Dict):
        """Log input data for reproducibility"""
        h = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        mlflow.log_param("input_hash", h)
        self._artifact(data, "input.json", "inputs")
        return self
    
    def log_output(self, data: Dict):
        """Log output for audit trail"""
        self._artifact(data, "output.json", "outputs")
        return self
    
    def log_lineage(self, source: str, transform: str):
        """Log data lineage (No Data Without Lineage principle)"""
        lineage = {
            "source": source,
            "transformation": transform,
            "agent": self._agent,
            "model": "claude-sonnet-4-20250514",
            "timestamp": datetime.utcnow().isoformat(),
            "session": self.session
        }
        mlflow.log_param("lineage_src", source)
        mlflow.log_param("lineage_xform", transform)
        self._artifact(lineage, "lineage.json", "governance")
        return self
    
    def _artifact(self, data: Any, name: str, folder: str):
        """Save JSON artifact"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            path = f.name
        mlflow.log_artifact(path, folder)
        os.unlink(path)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def track(agent: str, company: str, user: str = "system") -> ESGTracker:
    """Quick function to start tracking"""
    return ESGTracker().start(agent, company, user)


def quick_log(agent: str, company: str, score: int, 
              sector: str = None, confidence: float = 0.85) -> str:
    """One-liner to log a score. Returns run_id."""
    with track(agent, company) as t:
        t.log_score(score).log_confidence(confidence)
        if sector:
            t.log_sector(sector)
        return mlflow.active_run().info.run_id


def setup_experiments():
    """Create MLflow experiments for all 8 agents - run once"""
    mlflow.set_tracking_uri(MLFLOW_URI)
    print("=" * 50)
    print("Creating experiments for 8 ESG Navigator agents")
    print("=" * 50)
    
    for key, info in AGENTS.items():
        try:
            mlflow.set_experiment(f"TIS_ESG_Navigator/{key}")
            print(f"  ✅ {info['name']}")
        except Exception as e:
            print(f"  ⚠️ {info['name']}: {e}")
    
    print(f"\n✅ Done! View at: {MLFLOW_URI}")


def test_with_companies():
    """Test tracking with your 5 sample companies"""
    print("\n" + "=" * 50)
    print("Testing with ESG Navigator company data")
    print("=" * 50 + "\n")
    
    companies = [
        ("Ivanhoe Mines Ltd", "Mining", 78),
        ("Anglo American Platinum", "Mining", 81),
        ("Tiger Brands Limited", "Food", 45),
        ("Sasol Limited", "Chemicals", 58),
        ("Standard Bank Group", "Financial", 82),
    ]
    
    for name, sector, score in companies:
        with track("ESG_COMPLIANCE", name) as t:
            t.log_score(score)
            t.log_sector(sector)
            t.log_confidence(0.85)
            t.log_standards(["GRI", "SASB", "TCFD", "JSE"])
            t.log_lineage("neon_postgresql", "test_verification")
        
        arch = "SOPHISTICATED" if score >= 75 else "OPERATIONAL" if score >= 50 else "TRUE_LAGGARD"
        print(f"  ✅ {name}: {score} ({arch})")
    
    print(f"\n🎉 SUCCESS! View at: {MLFLOW_URI}")


# =============================================================================
# MAIN - RUN SETUP & TEST
# =============================================================================

if __name__ == "__main__":
    setup_experiments()
    test_with_companies()
