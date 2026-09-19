import { AlertTriangle, CalendarDays, Check, Database, Pencil, Search, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { fetchHistory, updateHistoryMetadata } from "./api";
import { chronological, finding, taskLabels, trendValue } from "./history";
import type { AnalysisMetadata, HistoryFilters, HistoryRecord, TaskId } from "./types";

function localDateTime(iso: string): string {
  const date = new Date(iso);
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 16);
}

function Sparkline({ records, unit }: { records: HistoryRecord[]; unit: string }) {
  const points = chronological(records)
    .map((record) => ({ record, value: trendValue(record) }))
    .filter((point): point is { record: HistoryRecord; value: number } => point.value !== null && Number.isFinite(point.value));
  if (!points.length) return <p className="empty-trend">No numeric trend is available.</p>;
  const values = points.map((point) => point.value);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const spread = Math.max(high - low, Math.abs(high) * 0.1, 0.01);
  const coordinates = points.map((point, index) => {
    const x = points.length === 1 ? 280 : 24 + (index / (points.length - 1)) * 512;
    const y = 136 - ((point.value - low) / spread) * 104;
    return { ...point, x, y };
  });
  return (
    <div className="sparkline-wrap">
      <svg viewBox="0 0 560 160" role="img" aria-label={`Trend from ${low.toFixed(3)} to ${high.toFixed(3)} ${unit}`}>
        <line x1="24" y1="136" x2="536" y2="136" className="chart-axis" />
        {coordinates.length > 1 && <polyline points={coordinates.map((point) => `${point.x},${point.y}`).join(" ")} className="chart-line" />}
        {coordinates.map((point) => <circle key={point.record.id} cx={point.x} cy={point.y} r="5" className={`chart-point ${point.record.mode}`}><title>{`${new Date(point.record.measurement_time).toLocaleString()}: ${point.value.toFixed(3)} ${unit}`}</title></circle>)}
      </svg>
      <div className="chart-range"><span>{new Date(points[0].record.measurement_time).toLocaleDateString()}</span><strong>Latest {points.at(-1)?.value.toFixed(3)} {unit}</strong><span>{new Date(points.at(-1)!.record.measurement_time).toLocaleDateString()}</span></div>
    </div>
  );
}

function CategoricalTrend({ records, task }: { records: HistoryRecord[]; task: "acv" | "corrugation" }) {
  const ordered = chronological(records);
  if (!ordered.length) return <p className="empty-trend">No categorical trend is available.</p>;
  const latest = task === "acv" ? String(ordered.at(-1)?.result.summary.top_car) : String(ordered.at(-1)?.result.summary.prediction);
  const repeated = ordered.filter((record) => String(task === "acv" ? record.result.summary.top_car : record.result.summary.prediction) === latest).length;
  return (
    <div>
      <div className="category-events">{ordered.map((record) => {
        const label = task === "acv" ? `Car ${record.result.summary.top_car}` : String(record.result.summary.prediction);
        return <span key={record.id} className={`category-event ${record.mode}`} title={new Date(record.measurement_time).toLocaleString()}>{label}</span>;
      })}</div>
      <p className="stability-copy">Latest result <strong>{task === "acv" ? `Car ${latest}` : latest}</strong> appears in {repeated} of {ordered.length} saved analyses for this asset.</p>
    </div>
  );
}

function TrendCard({ task, records }: { task: TaskId; records: HistoryRecord[] }) {
  return (
    <article className="trend-card">
      <div className="trend-heading"><div><span>{taskLabels[task]}</span><strong>{records.length} analysis{records.length === 1 ? "" : "es"}</strong></div>{records.some((record) => record.mode === "demo") && <small>Includes demo</small>}</div>
      {task === "door" && <Sparkline records={records} unit="% abnormal" />}
      {task === "shm" && <Sparkline records={records} unit="damage" />}
      {task === "acv" && <CategoricalTrend records={records} task="acv" />}
      {task === "corrugation" && <CategoricalTrend records={records} task="corrugation" />}
    </article>
  );
}

function HistoryDetail({ record, onClose, onUpdated }: { record: HistoryRecord; onClose: () => void; onUpdated: (record: HistoryRecord) => void }) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [draft, setDraft] = useState({ asset_id: record.asset_id, component_info: record.component_info, measurement_time: localDateTime(record.measurement_time) });
  const columns = Array.from(new Set(record.result.rows.flatMap((row) => Object.keys(row))));

  async function saveMetadata() {
    const payload: AnalysisMetadata = { asset_id: draft.asset_id.trim(), component_info: draft.component_info.trim(), measurement_time: new Date(draft.measurement_time).toISOString() };
    setSaving(true);
    try {
      const updated = await updateHistoryMetadata(record.id, payload);
      onUpdated(updated);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="history-detail-backdrop">
      <section className="history-detail" role="dialog" aria-modal="true" aria-labelledby="history-detail-title">
        <button className="dialog-close" onClick={onClose} aria-label="Close history detail"><X size={18} /></button>
        <div className="history-detail-heading"><span className={`mode-badge ${record.mode}`}>{record.mode}</span><h2 id="history-detail-title">{taskLabels[record.task]} analysis</h2><p>{finding(record)}</p></div>
        <section className="detail-section">
          <div className="section-heading"><div><span className="eyebrow">Asset metadata</span><h3>Saved context</h3></div><button className="text-button" onClick={() => setEditing((value) => !value)}><Pencil size={15} /> {editing ? "Cancel edit" : "Edit metadata"}</button></div>
          {editing ? <div className="metadata-grid"><label>Asset ID<input value={draft.asset_id} onChange={(event) => setDraft({ ...draft, asset_id: event.target.value })} /></label><label>Measurement time<input type="datetime-local" value={draft.measurement_time} onChange={(event) => setDraft({ ...draft, measurement_time: event.target.value })} /></label><label>Component / location<input value={draft.component_info} onChange={(event) => setDraft({ ...draft, component_info: event.target.value })} /></label><button className="primary-button" disabled={!draft.asset_id.trim() || !draft.measurement_time || saving} onClick={() => void saveMetadata()}>{saving ? "Saving…" : "Save metadata"}</button></div> : <dl className="detail-list"><div><dt>Asset</dt><dd>{record.asset_id}</dd></div><div><dt>Component / location</dt><dd>{record.component_info || "Not specified"}</dd></div><div><dt>Measurement</dt><dd>{new Date(record.measurement_time).toLocaleString()}</dd></div><div><dt>Analysed</dt><dd>{new Date(record.analysis_time).toLocaleString()}</dd></div></dl>}
        </section>
        <section className="detail-section immutable-section">
          <div className="section-heading"><div><span className="eyebrow">Immutable result</span><h3>Prediction record</h3></div><span className="locked-label"><Check size={14} /> Read only</span></div>
          <dl className="detail-list"><div><dt>Source</dt><dd>{record.source_file}</dd></div><div><dt>Model</dt><dd>{record.model_version}</dd></div><div><dt>SHA-256</dt><dd className="hash-value">{record.input_sha256}</dd></div><div><dt>Finding</dt><dd>{finding(record)}</dd></div></dl>
          <div className="table-wrap history-output"><table><thead><tr>{columns.map((column) => <th key={column}>{column.replaceAll("_", " ")}</th>)}</tr></thead><tbody>{record.result.rows.map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}</tr>)}</tbody></table></div>
        </section>
      </section>
    </div>
  );
}

export default function HistoryPage({ refreshKey = 0 }: { refreshKey?: number }) {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [filters, setFilters] = useState<HistoryFilters>({});
  const [appliedFilters, setAppliedFilters] = useState<HistoryFilters>({});
  const [focusAsset, setFocusAsset] = useState("");
  const [selected, setSelected] = useState<HistoryRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    void fetchHistory(appliedFilters).then((items) => {
      setRecords(items);
      setFocusAsset((current) => current && items.some((item) => item.asset_id === current) ? current : items[0]?.asset_id ?? "");
    }).catch(() => setError("History could not be loaded from the local database.")).finally(() => setLoading(false));
  }, [appliedFilters, refreshKey]);

  const assets = useMemo(() => Array.from(new Set(records.map((record) => record.asset_id))).sort(), [records]);
  const assetRecords = records.filter((record) => record.asset_id === focusAsset);
  const taskGroups = (["door", "acv", "corrugation", "shm"] as TaskId[]).map((task) => ({ task, records: assetRecords.filter((record) => record.task === task) })).filter((group) => group.records.length);
  const latest = chronological(assetRecords).at(-1);

  function updatedRecord(updated: HistoryRecord) {
    setRecords((current) => current.map((record) => record.id === updated.id ? updated : record));
    setSelected(updated);
    setFocusAsset(updated.asset_id);
  }

  function applyFilters() {
    setLoading(true);
    setError("");
    setAppliedFilters({ ...filters });
  }

  return (
    <section className="history-page">
      <div className="history-hero"><div><span className="eyebrow">Local operator record</span><h1>Analysis history</h1><p>Review asset trends, find previous results, and correct metadata without changing model predictions.</p></div><div className="database-pill"><Database size={18} /><div><strong>SQLite</strong><span>Stored on this laptop</span></div></div></div>
      <section className="history-filters">
        <label><Search size={16} /><input placeholder="Search asset ID" value={filters.asset ?? ""} onChange={(event) => setFilters({ ...filters, asset: event.target.value })} /></label>
        <select value={filters.task ?? ""} onChange={(event) => setFilters({ ...filters, task: event.target.value as TaskId | "" })}><option value="">All subsystems</option><option value="door">Door</option><option value="acv">ACV</option><option value="corrugation">Rail</option><option value="shm">SHM</option></select>
        <label><CalendarDays size={16} /><input type="date" value={filters.date_from ?? ""} onChange={(event) => setFilters({ ...filters, date_from: event.target.value })} aria-label="From date" /></label>
        <label><CalendarDays size={16} /><input type="date" value={filters.date_to ?? ""} onChange={(event) => setFilters({ ...filters, date_to: event.target.value })} aria-label="To date" /></label>
        <button className="primary-button" onClick={applyFilters}>Apply filters</button>
      </section>
      {error && <div className="error-banner"><AlertTriangle size={18} />{error}</div>}
      {loading ? <div className="history-empty">Loading history…</div> : records.length === 0 ? <div className="history-empty"><Database size={28} /><h2>No saved analyses yet</h2><p>Run an analysis, review its asset metadata, and confirm Save to begin an asset history.</p></div> : <>
        <section className="asset-summary-card">
          <div><span className="eyebrow">Asset summary</span><select value={focusAsset} onChange={(event) => setFocusAsset(event.target.value)} aria-label="Select asset for trends">{assets.map((asset) => <option key={asset}>{asset}</option>)}</select></div>
          <dl><div><dt>Saved analyses</dt><dd>{assetRecords.length}</dd></div><div><dt>Components</dt><dd>{new Set(assetRecords.map((record) => record.component_info).filter(Boolean)).size || "—"}</dd></div><div><dt>Latest measurement</dt><dd>{latest ? new Date(latest.measurement_time).toLocaleDateString() : "—"}</dd></div><div><dt>Demo records</dt><dd>{assetRecords.filter((record) => record.mode === "demo").length}</dd></div></dl>
        </section>
        <section className="trend-section"><div className="section-heading"><div><span className="eyebrow">Asset trends</span><h2>{focusAsset}</h2></div><p>Demo points are outlined and remain separate from operational evidence.</p></div><div className="trend-grid">{taskGroups.map((group) => <TrendCard key={group.task} task={group.task} records={group.records} />)}</div></section>
        <section className="history-table-card"><div className="section-heading"><div><span className="eyebrow">Chronological record</span><h2>Saved analyses</h2></div><span>{records.length} result{records.length === 1 ? "" : "s"}</span></div><div className="table-wrap"><table><thead><tr><th>Measurement</th><th>Asset</th><th>Component / location</th><th>Subsystem</th><th>Finding</th><th>Mode</th></tr></thead><tbody>{records.map((record) => <tr key={record.id} className="history-row" onClick={() => setSelected(record)}><td>{new Date(record.measurement_time).toLocaleString()}</td><td><strong>{record.asset_id}</strong></td><td>{record.component_info || "—"}</td><td>{taskLabels[record.task]}</td><td>{finding(record)}</td><td><span className={`mode-badge ${record.mode}`}>{record.mode}</span></td></tr>)}</tbody></table></div></section>
      </>}
      {selected && <HistoryDetail record={selected} onClose={() => setSelected(null)} onUpdated={updatedRecord} />}
    </section>
  );
}
