import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  Check,
  ChevronRight,
  CircleDot,
  DoorOpen,
  Fan,
  FileCheck2,
  Gauge,
  Menu,
  ShieldCheck,
  Sparkles,
  TrainFront,
  UploadCloud,
  Waves,
  X,
} from "lucide-react";
import { fetchTasks, runPrediction } from "./api";
import type { PredictionResponse, RunMode, TaskDescriptor, TaskId } from "./types";

const fallbackTasks: TaskDescriptor[] = [
  { id: "door", name: "Door diagnostics", short_name: "Door", description: "Detect door cycles and classify abnormal resistance.", accepted_extensions: [".csv"], output_filename: "door_predictions.csv", bundle_available: false },
  { id: "acv", name: "ACV leak localisation", short_name: "ACV", description: "Rank train cars by likelihood of a refrigerant leak.", accepted_extensions: [".xlsx"], output_filename: "acv_predictions.csv", bundle_available: false },
  { id: "corrugation", name: "Rail corrugation", short_name: "Rail", description: "Classify Normal, Side I, or Side II corrugation.", accepted_extensions: [".csv"], output_filename: "rail_predictions.csv", bundle_available: false },
  { id: "shm", name: "Structural health monitoring", short_name: "SHM", description: "Estimate cumulative fatigue damage from stress history.", accepted_extensions: [".csv", ".txt"], output_filename: "shm_predictions.csv", bundle_available: false },
];

const taskIcons = {
  door: DoorOpen,
  acv: Fan,
  corrugation: Waves,
  shm: Activity,
};

function formatNumber(value: unknown, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : String(value ?? "—");
}

function downloadText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function TopBar({ menuOpen, onToggle, connected }: { menuOpen: boolean; onToggle: () => void; connected: boolean }) {
  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onToggle} aria-label={menuOpen ? "Close menu" : "Open menu"}>
        {menuOpen ? <X size={20} /> : <Menu size={20} />}
      </button>
      <div className="brand-mark"><TrainFront size={20} strokeWidth={1.8} /></div>
      <div className="brand-copy"><strong>RailGuard</strong><span>Condition Intelligence</span></div>
      <div className="topbar-divider" />
      <span className="workspace-label">Analysis workspace</span>
      <div className="topbar-actions">
        <span className="system-pill"><span className={`live-dot ${connected ? "" : "offline"}`} /> API {connected ? "connected" : "offline"}</span>
        <span className="event-label">NebulaX 2026</span>
      </div>
    </header>
  );
}

function Sidebar({ tasks, selected, onSelect, open }: { tasks: TaskDescriptor[]; selected: TaskId; onSelect: (task: TaskId) => void; open: boolean }) {
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
      <div className="side-section-label">Monitor</div>
      <button className="side-link side-link-active"><Gauge size={18} /><span>New analysis</span></button>
      <div className="side-section-label side-gap">Subsystems</div>
      {tasks.map((task) => {
        const Icon = taskIcons[task.id];
        return (
          <button key={task.id} className={`side-link ${selected === task.id ? "side-current" : ""}`} onClick={() => onSelect(task.id)}>
            <Icon size={18} /><span>{task.short_name}</span>
            <span className={`availability-dot ${task.bundle_available ? "ready" : "demo"}`} title={task.bundle_available ? "Bundle ready" : "Demo fallback"} />
          </button>
        );
      })}
      <div className="sidebar-bottom">
        <div className="security-card"><ShieldCheck size={18} /><div><strong>Local inference</strong><span>Files stay on this machine</span></div></div>
        <p>RailGuard v1.0<br />Decision support only</p>
      </div>
    </aside>
  );
}

function TaskSelector({ tasks, selected, onSelect }: { tasks: TaskDescriptor[]; selected: TaskId; onSelect: (task: TaskId) => void }) {
  return (
    <section className="task-grid" aria-label="Select subsystem">
      {tasks.map((task, index) => {
        const Icon = taskIcons[task.id];
        return (
          <button className={`task-card ${selected === task.id ? "selected" : ""}`} onClick={() => onSelect(task.id)} key={task.id}>
            <span className="task-number">0{index + 1}</span>
            <span className="task-icon"><Icon size={22} /></span>
            <span className="task-text"><strong>{task.short_name}</strong><small>{task.description}</small></span>
            {selected === task.id && <span className="selected-check"><Check size={14} /></span>}
          </button>
        );
      })}
    </section>
  );
}

function UploadPanel({ task, file, onFile, onRun, running, mode, setMode }: {
  task: TaskDescriptor; file: File | null; onFile: (file: File | null) => void; onRun: () => void; running: boolean; mode: RunMode; setMode: (mode: RunMode) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const accept = task.accepted_extensions.join(",");

  function receive(files: FileList | null) {
    const next = files?.[0];
    if (next) onFile(next);
  }

  return (
    <section className="upload-card">
      <div className="card-heading">
        <div><span className="eyebrow">Input</span><h2>Upload sensor data</h2></div>
        <div className="mode-control" aria-label="Inference mode">
          {(["auto", "real", "demo"] as RunMode[]).map((value) => <button key={value} className={mode === value ? "active" : ""} onClick={() => setMode(value)}>{value}</button>)}
        </div>
      </div>
      {!file ? (
        <div
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); setDragging(false); receive(event.dataTransfer.files); }}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
        >
          <input ref={inputRef} type="file" accept={accept} hidden onChange={(event) => receive(event.target.files)} />
          <span className="upload-icon"><UploadCloud size={28} /></span>
          <strong>Drop your {task.short_name} file here</strong>
          <span>or click to browse · {task.accepted_extensions.join(" / ")} · up to 100 MB</span>
          <button className="secondary-button" type="button">Choose file</button>
        </div>
      ) : (
        <div className="file-ready">
          <span className="file-icon"><FileCheck2 size={24} /></span>
          <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · ready for {task.short_name} analysis</span></div>
          <button className="icon-button" onClick={() => onFile(null)} aria-label="Remove file"><X size={18} /></button>
        </div>
      )}
      <div className="upload-footer">
        <div className="bundle-status">
          <span className={`status-icon ${task.bundle_available ? "real" : "demo"}`}>{task.bundle_available ? <Check size={14} /> : <Sparkles size={14} />}</span>
          <div><strong>{task.bundle_available ? "Trusted bundle detected" : "Demo fallback available"}</strong><span>{task.bundle_available ? "Auto mode will run real inference." : "Configure a bundle for submission-ready output."}</span></div>
        </div>
        <button className="primary-button" disabled={!file || running} onClick={onRun}>
          {running ? <><span className="spinner" /> Analysing</> : <>Run analysis <ChevronRight size={17} /></>}
        </button>
      </div>
    </section>
  );
}

function ACVResult({ result }: { result: PredictionResponse }) {
  const ranking = result.visual.ranking ?? [];
  const max = Math.max(...ranking.map((item) => item.score), 0.001);
  return (
    <div className="result-layout">
      <div className="result-callout">
        <span className="callout-kicker">Most likely fault location</span>
        <div className="car-orbit"><span>CAR</span><strong>{String(result.summary.top_car).padStart(2, "0")}</strong></div>
        <p>Ranked first among {result.summary.cars_ranked} cars in the uploaded case.</p>
      </div>
      <div className="ranking-panel">
        <div className="panel-title"><strong>Fault ranking</strong><span>Relative model score</span></div>
        <div className="rank-bars">
          {ranking.map((item, index) => (
            <div className="rank-row" key={item.car}>
              <span className="rank-index">{index + 1}</span><span className="rank-car">Car {item.car}</span>
              <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.max(5, item.score / max * 100)}%` }} /></span>
              <span className="rank-score">{item.score.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DoorResult({ result }: { result: PredictionResponse }) {
  const segments = result.visual.segments ?? [];
  return (
    <div className="door-result">
      <div className="metric-row"><Metric label="Detected cycles" value={String(result.summary.cycles)} /><Metric label="Abnormal resistance" value={String(result.summary.abnormal_cycles)} alert={Number(result.summary.abnormal_cycles) > 0} /></div>
      <div className="table-wrap"><table><thead><tr><th>Start</th><th>End</th><th>Condition</th></tr></thead><tbody>{segments.map((row, index) => <tr key={index}><td>{row.start_time}</td><td>{row.end_time}</td><td><span className={`condition ${row.prediction === "Normal" ? "normal" : "alert"}`}>{row.prediction}</span></td></tr>)}</tbody></table></div>
    </div>
  );
}

function CorrugationResult({ result }: { result: PredictionResponse }) {
  const probabilities = result.visual.probabilities ?? {};
  return (
    <div className="classification-result">
      <div className="result-callout compact"><span className="callout-kicker">Track condition</span><strong className="class-label">{result.summary.prediction}</strong><p>{formatNumber(Number(result.summary.confidence) * 100, 1)}% model confidence</p></div>
      <div className="ranking-panel"><div className="panel-title"><strong>Class distribution</strong><span>Model probability</span></div>{Object.entries(probabilities).map(([label, value]) => <div className="probability-row" key={label}><span>{label}</span><span className="bar-track"><span className="bar-fill coral" style={{ width: `${value * 100}%` }} /></span><strong>{(value * 100).toFixed(1)}%</strong></div>)}</div>
    </div>
  );
}

function SHMResult({ result }: { result: PredictionResponse }) {
  const damage = Number(result.summary.predicted_damage);
  const interval = result.visual.interval;
  const gaugeValue = Math.min(100, Math.max(0, damage * 100));
  return (
    <div className="shm-result">
      <div className="damage-gauge" style={{ "--gauge": `${gaugeValue * 3.6}deg` } as CSSProperties}><div><span>Estimated damage</span><strong>{damage.toFixed(4)}</strong></div></div>
      <div className="damage-copy"><span className="eyebrow">Cumulative fatigue</span><h3>Structural damage estimate</h3><p>The prediction is produced from the uploaded dynamic-stress history.</p>{interval && <div className="interval"><span>Indicative interval</span><strong>{interval[0].toFixed(4)} – {interval[1].toFixed(4)}</strong></div>}</div>
    </div>
  );
}

function Metric({ label, value, alert = false }: { label: string; value: string; alert?: boolean }) {
  return <div className={`metric ${alert ? "metric-alert" : ""}`}><span>{label}</span><strong>{value}</strong></div>;
}

function ResultPanel({ result, onReset }: { result: PredictionResponse; onReset: () => void }) {
  return (
    <section className="results-card">
      <div className="results-header">
        <div><span className="eyebrow">Analysis complete</span><h2>{result.task_name}</h2><p>{result.source_file}</p></div>
        <div className="results-actions"><span className={`mode-badge ${result.mode}`}>{result.mode === "real" ? "Real inference" : "Demo result"}</span><button className="download-button" onClick={() => downloadText(result.output_filename, result.csv_text)}><ArrowDownToLine size={17} /> Download CSV</button></div>
      </div>
      {result.mode === "demo" && <div className="demo-banner"><AlertTriangle size={18} /><div><strong>Demonstration mode</strong><span>This output is simulated and must not be submitted for scoring.</span></div></div>}
      {result.task === "acv" && <ACVResult result={result} />}
      {result.task === "door" && <DoorResult result={result} />}
      {result.task === "corrugation" && <CorrugationResult result={result} />}
      {result.task === "shm" && <SHMResult result={result} />}
      <div className="results-footer"><div><CircleDot size={15} /><span>Output validated against the official {result.output_filename} schema.</span></div><button className="text-button" onClick={onReset}>Start another analysis</button></div>
    </section>
  );
}

export default function App() {
  const [tasks, setTasks] = useState<TaskDescriptor[]>(fallbackTasks);
  const [selected, setSelected] = useState<TaskId>("acv");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<RunMode>("auto");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);

  useEffect(() => {
    fetchTasks()
      .then((nextTasks) => { setTasks(nextTasks); setApiConnected(true); })
      .catch(() => { setApiConnected(false); setError("The API is unavailable. Start the FastAPI service to run analysis."); });
  }, []);
  const task = useMemo(() => tasks.find((item) => item.id === selected) ?? tasks[0], [tasks, selected]);

  function selectTask(next: TaskId) { setSelected(next); setFile(null); setResult(null); setError(""); setMenuOpen(false); }
  async function analyse() {
    if (!file) return;
    setRunning(true); setError(""); setResult(null);
    try { setResult(await runPrediction(selected, file, mode)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Analysis failed"); }
    finally { setRunning(false); }
  }

  return (
    <div className="app-shell">
      <TopBar menuOpen={menuOpen} onToggle={() => setMenuOpen((value) => !value)} connected={apiConnected} />
      <Sidebar tasks={tasks} selected={selected} onSelect={selectTask} open={menuOpen} />
      <main className="main-content">
        <section className="intro">
          <div><span className="eyebrow">Condition monitoring console</span><h1>Turn sensor data into<br /><em>maintenance decisions.</em></h1><p>Upload rail telemetry, run a trusted model, and export competition-ready predictions without touching code.</p></div>
          <div className="coverage-stat"><span>SUBSYSTEM COVERAGE</span><strong>04<small>/04</small></strong><div><span className="coverage-line" /><p>Door · ACV · Rail · SHM</p></div></div>
        </section>
        <div className="workflow-strip"><span className="workflow-active"><b>1</b> Select subsystem</span><i /><span className={file ? "workflow-active" : ""}><b>2</b> Upload data</span><i /><span className={result ? "workflow-active" : ""}><b>3</b> Review result</span><i /><span className={result ? "workflow-active" : ""}><b>4</b> Download</span></div>
        <TaskSelector tasks={tasks} selected={selected} onSelect={selectTask} />
        {error && <div className="error-banner"><AlertTriangle size={18} /><span>{error}</span><button onClick={() => setError("")}><X size={16} /></button></div>}
        {!result ? <UploadPanel task={task} file={file} onFile={setFile} onRun={analyse} running={running} mode={mode} setMode={setMode} /> : <ResultPanel result={result} onReset={() => { setResult(null); setFile(null); }} />}
        <footer><span>RailGuard AI</span><p>Evidence for operators. Predictions for maintenance teams.</p><p>Not an approved maintenance rule.</p></footer>
      </main>
    </div>
  );
}
