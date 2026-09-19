import { AlertTriangle, Check, Database, Save, X } from "lucide-react";
import { useState } from "react";
import { saveAnalysis } from "./api";
import type { AnalysisMetadata, HistoryRecord, PredictionResponse } from "./types";

export interface MetadataDraft {
  asset_id: string;
  component_info: string;
  measurement_time: string;
}

function metadataPayload(metadata: MetadataDraft): AnalysisMetadata {
  return {
    asset_id: metadata.asset_id.trim(),
    component_info: metadata.component_info.trim(),
    measurement_time: new Date(metadata.measurement_time).toISOString(),
  };
}

export function SaveAnalysisPanel({
  result,
  metadata,
  onMetadata,
  saved,
  onSaved,
  onOpenHistory,
}: {
  result: PredictionResponse;
  metadata: MetadataDraft;
  onMetadata: (metadata: MetadataDraft) => void;
  saved: HistoryRecord | null;
  onSaved: (record: HistoryRecord) => void;
  onOpenHistory: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const valid = Boolean(metadata.asset_id.trim() && metadata.measurement_time);

  async function confirmSave() {
    if (!valid || saving) return;
    setSaving(true);
    setError("");
    try {
      onSaved(await saveAnalysis(metadataPayload(metadata), result));
      setConfirming(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The analysis could not be saved.");
    } finally {
      setSaving(false);
    }
  }

  if (saved) {
    return (
      <section className="save-card saved-card">
        <span className="save-icon complete"><Check size={19} /></span>
        <div><span className="eyebrow">Saved locally</span><h3>Analysis added to {saved.asset_id}</h3><p>The prediction is now immutable. Asset metadata can still be corrected from History.</p></div>
        <button className="secondary-button" onClick={onOpenHistory}>Open history</button>
      </section>
    );
  }

  return (
    <section className="save-card">
      <span className="save-icon"><Database size={20} /></span>
      <div className="save-card-body">
        <span className="eyebrow">Analysis history</span>
        <h3>Save this result to the asset record</h3>
        <p>Confirm the asset context before writing to the local SQLite history.</p>
        <div className="metadata-grid compact">
          <label>Asset ID<input value={metadata.asset_id} onChange={(event) => onMetadata({ ...metadata, asset_id: event.target.value })} required /></label>
          <label>Measurement date and time<input type="datetime-local" value={metadata.measurement_time} onChange={(event) => onMetadata({ ...metadata, measurement_time: event.target.value })} required /></label>
          <label>Component or location<input value={metadata.component_info} onChange={(event) => onMetadata({ ...metadata, component_info: event.target.value })} placeholder="Optional, e.g. Door 2L" /></label>
        </div>
        {error && <div className="inline-error"><AlertTriangle size={16} />{error}</div>}
      </div>
      <button className="primary-button" disabled={!valid} onClick={() => setConfirming(true)}><Save size={16} /> Review and save</button>
      {confirming && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirmation-dialog" role="dialog" aria-modal="true" aria-labelledby="save-confirmation-title">
            <button className="dialog-close" onClick={() => setConfirming(false)} aria-label="Close confirmation"><X size={18} /></button>
            <span className="eyebrow">Confirm metadata</span>
            <h2 id="save-confirmation-title">Save analysis to history?</h2>
            <p>Check these details carefully. The prediction will become read-only after saving.</p>
            <dl>
              <div><dt>Asset</dt><dd>{metadata.asset_id.trim()}</dd></div>
              <div><dt>Component / location</dt><dd>{metadata.component_info.trim() || "Not specified"}</dd></div>
              <div><dt>Measurement time</dt><dd>{new Date(metadata.measurement_time).toLocaleString()}</dd></div>
              <div><dt>Subsystem</dt><dd>{result.task_name}</dd></div>
              <div><dt>Result type</dt><dd>{result.mode === "demo" ? "Demonstration" : "Real model analysis"}</dd></div>
              <div><dt>Source file</dt><dd>{result.source_file}</dd></div>
            </dl>
            {result.mode === "demo" && <div className="demo-confirmation"><AlertTriangle size={17} />This simulated result will be clearly marked as Demo in history.</div>}
            <div className="dialog-actions"><button className="secondary-button" onClick={() => setConfirming(false)}>Go back and edit</button><button className="primary-button" disabled={saving} onClick={() => void confirmSave()}>{saving ? "Saving…" : "Confirm save"}</button></div>
          </section>
        </div>
      )}
    </section>
  );
}
