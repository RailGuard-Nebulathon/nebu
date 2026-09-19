import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  BookOpen,
  Check,
  ChevronRight,
  CircleHelp,
  DoorOpen,
  Eye,
  Fan,
  FileCheck2,
  Gauge,
  ListChecks,
  Menu,
  Plus,
  SlidersHorizontal,
  TrainFront,
  UploadCloud,
  Waves,
  X,
} from "lucide-react";
import { fetchTasks, runPrediction } from "./api";
import { buildDecision } from "./decision";
import type { DecisionViewModel } from "./decision";
import type { PredictionResponse, TaskDescriptor, TaskId } from "./types";
import { getUploadError, MAX_UPLOAD_LABEL } from "./upload";


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

function TopBar({ menuOpen, onToggle }: { menuOpen: boolean; onToggle: () => void }) {
  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onToggle} aria-label={menuOpen ? "Close menu" : "Open menu"}>
        {menuOpen ? <X size={20} /> : <Menu size={20} />}
      </button>
      <div className="brand-mark"><TrainFront size={20} strokeWidth={1.8} /></div>
      <div className="brand-copy"><strong>RailGuard</strong></div>
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
            <span className={`availability-dot ${task.bundle_available ? "ready" : "unavailable"}`} title={task.bundle_available ? "Analysis ready" : "Analysis unavailable"} />
          </button>
        );
      })}
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

function UploadPanel({ task, file, onFile, onRun, running, busy }: {
  task: TaskDescriptor; file: File | null; onFile: (file: File | null) => void; onRun: () => void; running: boolean; busy: boolean;
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
          <span>or click to browse · {task.accepted_extensions.join(" / ")} · up to {MAX_UPLOAD_LABEL}</span>
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
          <span className={`status-icon ${task.bundle_available ? "ready" : "unavailable"}`}>{task.bundle_available ? <Check size={14} /> : <AlertTriangle size={14} />}</span>
          <div><strong>{task.bundle_available ? "Analysis ready" : "Analysis unavailable"}</strong><span>{task.bundle_available ? "The approved model is available." : "This subsystem has not been configured yet."}</span></div>
        </div>
        <button className="primary-button" disabled={!file || busy || !task.bundle_available} onClick={onRun}>
          {running ? <><span className="spinner" /> Analysing</> : <>Run analysis <ChevronRight size={17} /></>}
        </button>
      </div>
    </section>
  );
}

function ACVResult({ result }: { result: PredictionResponse }) {
  const ranking = result.visual.ranking ?? [];
  const margin = ranking.length > 1 ? ranking[0].score - ranking[1].score : null;
  return (
    <div className="result-layout">
      <div className="result-callout">
        <span className="callout-kicker">Most likely fault location</span>
        <div className="car-orbit"><span>CAR</span><strong>{String(result.summary.top_car).padStart(2, "0")}</strong></div>
        <p>Ranked first among {result.summary.cars_ranked} cars in the uploaded case.</p>
        {margin !== null && <div className="margin-chip"><span>Top-two margin</span><strong>{margin.toFixed(2)}</strong></div>}
      </div>
      <div className="ranking-panel">
        <div className="panel-title"><strong>Fault ranking</strong><span>Model score · fixed 0–1 scale</span></div>
        <div className="rank-bars">
          {ranking.map((item, index) => (
            <div className="rank-row" key={item.car}>
              <span className="rank-index">{index + 1}</span><span className="rank-car">Car {item.car}</span>
              <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.max(2, Math.min(100, item.score * 100))}%` }} /></span>
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
      <div className="table-wrap"><table><thead><tr><th>Start</th><th>End</th><th>Condition</th><th>Model confidence</th></tr></thead><tbody>{segments.map((row, index) => <tr key={index}><td>{row.start_time}</td><td>{row.end_time}</td><td><span className={`condition ${row.prediction === "Normal" ? "normal" : "alert"}`}>{row.prediction}</span></td><td>{typeof row.confidence === "number" ? `${(row.confidence * 100).toFixed(1)}%` : "Not reported"}</td></tr>)}</tbody></table></div>
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
  return (
    <div className="shm-result">
      <div className="damage-estimate"><span>Model estimate</span><strong>{damage.toFixed(4)}</strong><small>Raw cumulative-damage value · not a percentage</small></div>
      <div className="damage-copy"><span className="eyebrow">Cumulative fatigue</span><h3>Structural damage estimate</h3><p>Interpret this screening estimate against approved engineering limits. RailGuard does not apply an invented pass/fail threshold.</p>{interval && <div className="interval"><span>Indicative interval</span><strong>{interval[0].toFixed(4)} – {interval[1].toFixed(4)}</strong></div>}</div>
    </div>
  );
}

function Metric({ label, value, alert = false }: { label: string; value: string; alert?: boolean }) {
  return <div className={`metric ${alert ? "metric-alert" : ""}`}><span>{label}</span><strong>{value}</strong></div>;
}

function TrainInspectionMap({ result }: { result: PredictionResponse }) {
  const ranking = result.visual.ranking ?? [];
  const orderedCars = [...ranking].sort((left, right) => left.car.localeCompare(right.car, undefined, { numeric: true }));
  const primary = String(result.summary.top_car ?? ranking[0]?.car ?? "");
  const secondary = ranking[1]?.car;
  return (
    <section className="train-map" aria-label="Train car inspection map">
      <div className="train-map-copy"><span>TRAIN VIEW</span><strong>Inspection order at a glance</strong><small>Highlighted by model rank, not physical severity.</small></div>
      <div className="train-cars">
        {orderedCars.map((item) => {
          const state = item.car === primary ? "primary" : item.car === secondary ? "secondary" : "unflagged";
          return <div className={`train-car ${state}`} data-status={state} key={item.car}><span>CAR</span><strong>{String(item.car).padStart(2, "0")}</strong></div>;
        })}
      </div>
      <div className="train-map-legend"><span><i className="primary" /> Inspect first</span><span><i className="secondary" /> Inspect next if inconclusive</span></div>
    </section>
  );
}

function QuickDecision({ result, decision }: { result: PredictionResponse; decision: DecisionViewModel }) {
  const tone = decision.status.toLowerCase().replaceAll(" ", "-");
  const reliabilityTone = decision.reliability.toLowerCase().replaceAll(" ", "-");
  return (
    <section className={`decision-card tone-${tone}`} aria-labelledby="decision-finding">
      <div className="decision-lead">
        <span className="decision-status"><span className="status-symbol" aria-hidden="true" />{decision.status}</span>
        <h3 id="decision-finding">{decision.finding}</h3>
        <p>{decision.urgency}</p>
      </div>
      <div className="decision-facts">
        <div><span>LOCATION</span><strong>{decision.location}</strong></div>
        <div className={`reliability reliability-${reliabilityTone}`}><span>RELIABILITY</span><strong>{decision.reliability}</strong><small>{decision.reliabilityReason}</small></div>
      </div>
      {result.task === "acv" && <TrainInspectionMap result={result} />}
      <div className="next-checks">
        <h4><ListChecks size={17} /> Next checks</h4>
        <ol>{decision.nextChecks.map((check) => <li key={check}>{check}</li>)}</ol>
      </div>
    </section>
  );
}

function WhyResult({ decision }: { decision: DecisionViewModel }) {
  return (
    <section className="explanation-panel">
      <div className="explanation-heading"><Eye size={20} /><div><span className="eyebrow">Plain-language evidence</span><h3>Why this result?</h3><p>These signals influenced the model output; they do not establish physical cause.</p></div></div>
      <div className="evidence-statements">{decision.evidence.map((item, index) => <div key={item}><span>0{index + 1}</span><p>{item}</p></div>)}</div>
      <div className="reliability-explainer"><strong>{decision.reliability}</strong><p>{decision.reliabilityReason}</p></div>
    </section>
  );
}

function TechnicalEvidence({ result }: { result: PredictionResponse }) {
  if (result.task === "acv") return <ACVResult result={result} />;
  if (result.task === "door") return <DoorResult result={result} />;
  if (result.task === "corrugation") return <CorrugationResult result={result} />;
  return <SHMResult result={result} />;
}

function LearnView({ task }: { task: TaskId }) {
  const taskGuide = task === "acv"
    ? { term: "Model score", meaning: "A ranking signal for each car. Compare cars and the top-two margin; do not read it as proof that a leak exists.", chart: "Longer bars indicate stronger model support on a fixed 0–1 scale." }
    : task === "door"
      ? { term: "Abnormal resistance", meaning: "A door cycle whose measured pattern was classified differently from learned normal operation.", chart: "Use the start and end times to locate flagged cycles in the source signals." }
      : task === "corrugation"
        ? { term: "Class probability", meaning: "Relative model support for Normal, Side I, and Side II—not the probability that operation is safe.", chart: "Compare the complete class distribution; closely matched bars require review." }
        : { term: "Prediction interval", meaning: "A range expressing model uncertainty around cumulative damage.", chart: "Compare the entire interval—not only the centre estimate—with approved engineering limits." };
  return (
    <section className="learn-panel">
      <div className="learn-heading"><BookOpen size={21} /><div><span className="eyebrow">Engineer onboarding</span><h3>How to read this result</h3></div></div>
      <div className="learn-grid">
        <article><span>KEY TERM</span><h4>{taskGuide.term}</h4><p>{taskGuide.meaning}</p></article>
        <article><span>READ THE CHART</span><h4>Interpretation</h4><p>{taskGuide.chart}</p></article>
        <article><span>SAFE USE</span><h4>Model evidence, not cause</h4><p>Important features and signal regions describe model influence. They do not prove a root cause or replace an approved inspection.</p></article>
        <article><span>NORMAL RANGES</span><h4>Use official limits only</h4><p>RailGuard shows a normal range only when one is supplied by an approved data source. No engineering limits are invented in the UI.</p></article>
      </div>
    </section>
  );
}

type ResultView = "quick" | "why" | "technical" | "learn";

export function ResultPanel({ result, onReset }: { result: PredictionResponse; onReset: () => void }) {
  const [view, setView] = useState<ResultView>("quick");
  const decision = buildDecision(result);
  const views: Array<{ id: ResultView; label: string; icon: typeof Eye }> = [
    { id: "quick", label: "Quick decision", icon: ListChecks },
    { id: "why", label: "Why this result?", icon: CircleHelp },
    { id: "technical", label: "Technical evidence", icon: SlidersHorizontal },
    { id: "learn", label: "Learn", icon: BookOpen },
  ];
  return (
    <section className="results-card">
      <div className="results-header">
        <div><span className="eyebrow">Analysis complete</span><h2>{result.task_name}</h2><p>{result.source_file}</p></div>
        <div className="results-actions"><span className={`mode-badge ${result.mode}`}>{result.mode === "real" ? "Analysis completed" : "Demonstration result"}</span><button className="new-analysis-button" onClick={onReset}><Plus size={17} /> Start another analysis</button><button className="download-button" onClick={() => downloadText(result.output_filename, result.csv_text)}><ArrowDownToLine size={17} /> Export data</button></div>
      </div>
      {result.mode === "demo" && <div className="demo-banner"><AlertTriangle size={18} /><div><strong>Demonstration mode</strong><span>This output is simulated and cannot support an operational decision or competition submission.</span></div></div>}
      <nav className="result-navigation" aria-label="Result detail level">
        {views.map(({ id, label, icon: Icon }) => <button key={id} className={view === id ? "active" : ""} aria-pressed={view === id} onClick={() => setView(id)}><Icon size={16} />{label}</button>)}
      </nav>
      {view === "quick" && <QuickDecision result={result} decision={decision} />}
      {view === "why" && <WhyResult decision={decision} />}
      {view === "technical" && <TechnicalEvidence result={result} />}
      {view === "learn" && <LearnView task={result.task} />}
    </section>
  );
}

export default function App() {
  const [tasks, setTasks] = useState<TaskDescriptor[]>(fallbackTasks);
  const [selected, setSelected] = useState<TaskId>("acv");
  const [files, setFiles] = useState<Partial<Record<TaskId, File | null>>>({});
  const [results, setResults] = useState<Partial<Record<TaskId, PredictionResponse | null>>>({});
  const [runningTask, setRunningTask] = useState<TaskId | null>(null);
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [retryAction, setRetryAction] = useState<"tasks" | "analysis" | null>(null);

  function retryLoadTasks() {
    setError("");
    setRetryAction(null);
    fetchTasks()
      .then(setTasks)
      .catch(() => {
        setRetryAction("tasks");
        setError("The analysis service is temporarily unavailable. Try again in a few minutes or contact the RailGuard administrator.");
      });
  }

  useEffect(() => {
    void fetchTasks()
      .then(setTasks)
      .catch(() => {
        setRetryAction("tasks");
        setError("The analysis service is temporarily unavailable. Try again in a few minutes or contact the RailGuard administrator.");
      });
  }, []);
  const task = useMemo(() => tasks.find((item) => item.id === selected) ?? tasks[0], [tasks, selected]);
  const file = files[selected] ?? null;
  const result = results[selected] ?? null;

  function selectTask(next: TaskId) { setSelected(next); setError(""); setRetryAction(null); setMenuOpen(false); }
  function selectFile(next: File | null) {
    const uploadError = next ? getUploadError(next) : null;
    if (uploadError) {
      setFiles((current) => ({ ...current, [selected]: null }));
      setRetryAction(null);
      setError(uploadError);
      return;
    }
    setFiles((current) => ({ ...current, [selected]: next }));
    setError("");
    setRetryAction(null);
  }
  async function analyse() {
    if (!file || runningTask !== null) return;
    const taskId = selected;
    setRunningTask(taskId); setError(""); setRetryAction(null);
    try {
      const nextResult = await runPrediction(taskId, file);
      setResults((current) => ({ ...current, [taskId]: nextResult }));
      setFiles((current) => ({ ...current, [taskId]: null }));
    }
    catch (caught) {
      setRetryAction("analysis");
      setError(caught instanceof Error ? caught.message : "The analysis could not be completed. Check the file and try again.");
    }
    finally { setRunningTask(null); }
  }

  function resetSelectedAnalysis() {
    setResults((current) => ({ ...current, [selected]: null }));
    setFiles((current) => ({ ...current, [selected]: null }));
    setError("");
    setRetryAction(null);
  }

  function retry() {
    if (retryAction === "tasks") retryLoadTasks();
    if (retryAction === "analysis") void analyse();
  }

  return (
    <div className="app-shell">
      <TopBar menuOpen={menuOpen} onToggle={() => setMenuOpen((value) => !value)} />
      <Sidebar tasks={tasks} selected={selected} onSelect={selectTask} open={menuOpen} />
      <main className="main-content">
        <section className="intro">
          <div><h1>Turn sensor data into<br /><em>maintenance decisions.</em></h1></div>
        </section>
        <TaskSelector tasks={tasks} selected={selected} onSelect={selectTask} />
        {error && <div className="error-banner" role="alert"><AlertTriangle size={18} /><span>{error}</span>{retryAction && <button className="retry-button" onClick={retry}>Try again</button>}<button className="dismiss-button" onClick={() => { setError(""); setRetryAction(null); }} aria-label="Dismiss message"><X size={16} /></button></div>}
        {!result ? <UploadPanel task={task} file={file} onFile={selectFile} onRun={analyse} running={runningTask === selected} busy={runningTask !== null} /> : <ResultPanel result={result} onReset={resetSelectedAnalysis} />}
      </main>
    </div>
  );
}
