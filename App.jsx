import { useState, useCallback } from "react";

// ── Config ────────────────────────────────────────────────────────────────────
// In production this is your Render URL.
// Locally set VITE_API_URL=http://localhost:8000 in .env.local
const API_URL = import.meta.env.VITE_API_URL || "https://tricab-rfq-api.onrender.com";

const CONF_COLOR = {
  HIGH:   { bg: "#d1fae5", text: "#065f46", border: "#6ee7b7" },
  MEDIUM: { bg: "#fef3c7", text: "#92400e", border: "#fcd34d" },
  LOW:    { bg: "#fee2e2", text: "#991b1b", border: "#fca5a5" },
  NONE:   { bg: "#f1f5f9", text: "#475569", border: "#cbd5e1" },
};

const CONF_UNDERLINE = {
  HIGH: "#10b981", MEDIUM: "#f59e0b", LOW: "#ef4444", NONE: "#94a3b8",
};

// ── API call ──────────────────────────────────────────────────────────────────
async function callAPI(text) {
  const res = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "API error");
  return data;
}

// ── Components ────────────────────────────────────────────────────────────────
function ConfidenceBadge({ level }) {
  const c = CONF_COLOR[level] || CONF_COLOR.NONE;
  return (
    <span style={{
      background: c.bg, color: c.text, border: `1px solid ${c.border}`,
      borderRadius: 4, padding: "1px 7px", fontSize: 12, fontWeight: 600,
      whiteSpace: "nowrap",
    }}>
      {level}
    </span>
  );
}

function SegmentTable({ segments, notes, missing, styleDescription }) {
  return (
    <div style={{ padding: "10px 0 4px" }}>
      {styleDescription && (
        <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 8, fontStyle: "italic" }}>
          {styleDescription}
        </div>
      )}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#f9fafb" }}>
            {["#", "Segment", "Value", "Display", "Confidence", "Note"].map(h => (
              <th key={h} style={{
                textAlign: "left", padding: "4px 8px",
                borderBottom: "1px solid #e5e7eb", fontWeight: 600, color: "#374151",
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {segments.map(s => (
            <tr key={s.position} style={{ borderBottom: "1px solid #f3f4f6" }}>
              <td style={{ padding: "3px 8px", color: "#9ca3af" }}>{s.position}</td>
              <td style={{ padding: "3px 8px", color: "#6b7280" }}>{s.label}</td>
              <td style={{ padding: "3px 8px", fontFamily: "monospace", fontWeight: 600 }}>{s.value}</td>
              <td style={{ padding: "3px 8px", color: "#374151" }}>{s.display}</td>
              <td style={{ padding: "3px 8px" }}><ConfidenceBadge level={s.confidence} /></td>
              <td style={{ padding: "3px 8px", color: "#6b7280", fontSize: 12 }}>{s.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {notes?.length > 0 && (
        <div style={{ marginTop: 8 }}>
          {notes.map((n, i) => (
            <div key={i} style={{ fontSize: 12, color: "#92400e", background: "#fef3c7",
              borderRadius: 4, padding: "4px 8px", marginBottom: 4 }}>
              ℹ️ {n}
            </div>
          ))}
        </div>
      )}
      {missing?.length > 0 && (
        <div style={{ marginTop: 6 }}>
          {missing.map((m, i) => (
            <div key={i} style={{ fontSize: 12, color: "#991b1b", background: "#fee2e2",
              borderRadius: 4, padding: "4px 8px", marginBottom: 4 }}>
              ⚠️ Missing: {m}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ResultRow({ requestText, result, error, expanded, onToggle }) {
  if (error) {
    return (
      <tr style={{ borderBottom: "1px solid #f3f4f6" }}>
        <td style={{ padding: "10px 12px", verticalAlign: "top", color: "#374151" }}>
          {requestText}
        </td>
        <td style={{ padding: "10px 12px", color: "#991b1b", fontSize: 13 }}>
          ⚠️ {error}
        </td>
      </tr>
    );
  }

  const conf = result.overall_confidence;
  const underline = CONF_UNDERLINE[conf] || CONF_UNDERLINE.NONE;

  return (
    <>
      <tr
        style={{ borderBottom: expanded ? "none" : "1px solid #f3f4f6", cursor: "pointer" }}
        onClick={onToggle}
      >
        <td style={{ padding: "10px 12px", verticalAlign: "top", color: "#374151", width: "45%" }}>
          {requestText}
        </td>
        <td style={{ padding: "10px 12px", verticalAlign: "top" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span style={{
              fontFamily: "monospace", fontWeight: 700, fontSize: 15,
              borderBottom: `2px solid ${underline}`,
            }}>
              {result.product_code}
            </span>
            <ConfidenceBadge level={conf} />
            {result.variant_exists && (
              <span style={{ fontSize: 11, color: "#065f46" }}>✓ in catalogue</span>
            )}
            <span style={{ marginLeft: "auto", color: "#9ca3af", fontSize: 13 }}>
              {expanded ? "▲" : "▼"}
            </span>
          </div>
        </td>
      </tr>
      {expanded && (
        <tr style={{ borderBottom: "1px solid #f3f4f6" }}>
          <td colSpan={2} style={{ padding: "0 12px 12px", background: "#fafafa" }}>
            <SegmentTable
              segments={result.segments}
              notes={result.notes}
              missing={result.missing}
              styleDescription={result.style_description}
            />
          </td>
        </tr>
      )}
    </>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [input, setInput]       = useState("");
  const [rows, setRows]         = useState([]);
  const [loading, setLoading]   = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [expanded, setExpanded] = useState({});

  const toggleExpanded = useCallback((idx) => {
    setExpanded(prev => ({ ...prev, [idx]: !prev[idx] }));
  }, []);

  const handleGenerate = async () => {
    const lines = input.split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;

    setLoading(true);
    setRows([]);
    setExpanded({});
    setProgress({ done: 0, total: lines.length });

    const results = new Array(lines.length).fill(null);

    await Promise.all(lines.map(async (line, idx) => {
      try {
        const data = await callAPI(line);
        results[idx] = { requestText: line, result: data };
      } catch (err) {
        results[idx] = { requestText: line, error: err.message };
      }
      setProgress(p => ({ ...p, done: p.done + 1 }));
    }));

    setRows(results);
    setLoading(false);
  };

  const progressPct = progress.total > 0
    ? Math.round((progress.done / progress.total) * 100)
    : 0;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 960, margin: "0 auto", padding: 24 }}>
      <h2 style={{ marginBottom: 4, color: "#111827" }}>TriCab RFQ Product Code Generator</h2>
      <p style={{ color: "#6b7280", marginBottom: 16, fontSize: 14 }}>
        One cable description per line. Click a result to expand segment details.
      </p>

      <textarea
        value={input}
        onChange={e => setInput(e.target.value)}
        placeholder={
          "3C 2.5mm flex control PVC black\n" +
          "single core 95mm XLPE copper orange\n" +
          "7 pair 1.5mm instrumentation SWA grey\n" +
          "FR 110° CU 4C+E 16mm2"
        }
        style={{
          width: "100%", height: 130, padding: 10, fontSize: 14,
          border: "1px solid #d1d5db", borderRadius: 6, resize: "vertical",
          fontFamily: "monospace", boxSizing: "border-box",
        }}
      />

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 10 }}>
        <button
          onClick={handleGenerate}
          disabled={loading || !input.trim()}
          style={{
            background: loading ? "#9ca3af" : "#2563eb", color: "#fff",
            border: "none", borderRadius: 6, padding: "8px 20px",
            fontSize: 14, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? `Generating… ${progress.done}/${progress.total}` : "Generate Codes"}
        </button>

        {loading && (
          <div style={{ flex: 1, height: 6, background: "#e5e7eb", borderRadius: 3 }}>
            <div style={{
              width: `${progressPct}%`, height: "100%",
              background: "#10b981", borderRadius: 3, transition: "width 0.2s",
            }} />
          </div>
        )}
      </div>

      {rows.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ background: "#f3f4f6" }}>
                <th style={{
                  textAlign: "left", padding: "10px 12px", fontWeight: 600,
                  color: "#374151", borderBottom: "2px solid #e5e7eb", width: "45%",
                }}>Request</th>
                <th style={{
                  textAlign: "left", padding: "10px 12px", fontWeight: 600,
                  color: "#374151", borderBottom: "2px solid #e5e7eb",
                }}>Product Code</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <ResultRow
                  key={idx}
                  requestText={row.requestText}
                  result={row.result}
                  error={row.error}
                  expanded={!!expanded[idx]}
                  onToggle={() => toggleExpanded(idx)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
