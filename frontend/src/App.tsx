import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Activity, 
  Dna, 
  Stethoscope, 
  User, 
  TrendingUp, 
  AlertCircle,
  RefreshCw,
  LayoutDashboard,
  Shield,
  Monitor,
  Cpu,
  ArrowUpRight
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar, Cell,
  Radar, RadarChart, PolarGrid, PolarAngleAxis
} from 'recharts';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Use environment variable in production, localhost in development
const API_BASE = import.meta.env.PROD 
  ? (import.meta.env.VITE_API_URL || window.location.origin)
  : "http://localhost:8000";

// --- Types ---

interface Field {
  name: string;
  label: string;
  type: "select" | "number";
  options?: (string | number)[];
  min?: number;
  max?: number;
}

interface Group {
  group: string;
  fields: Field[];
}

interface Mutation {
  name: string;
  determination: string;
}

interface Metadata {
  schema: Group[];
  mutations: Mutation[];
}

interface Result {
  probability: number;
  risk_level: string;
  mutation_impact: string | null;
}

interface DashboardData {
  summary: {
    final_accuracy: number;
    f1_score: number;
    num_clients: number;
    privacy_budget: number;
    precision?: number;
    recall?: number;
    specificity?: number;
    auc?: number;
    total_samples?: number;
  };
  training_progress: any[];
  data_distribution: any[];
  radar_metrics: any[];
  confusion_matrix: {
    tp: number; fp: number; fn: number; tn: number;
  };
  privacy_settings: {
    label: string;
    value: string;
    desc: string;
  }[];
}

// --- Main App ---

export default function App() {
  const [view, setView] = useState<'dashboard' | 'diagnose' | 'simulator'>('dashboard');
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        const [metaResp, dashResp] = await Promise.all([
          axios.get(`${API_BASE}/api/metadata`),
          axios.get(`${API_BASE}/api/dashboard-metrics`)
        ]);
        setMetadata(metaResp.data);
        setDashboardData(dashResp.data);
        setLoading(false);
      } catch (err: any) {
        setErrorMsg("Failed to connect to diagnostic server. " + (err.message || ""));
        setLoading(false);
      }
    };
    init();
  }, []);

  if (loading) return <LoadingScreen />;

  if (errorMsg) {
    return (
      <div className="min-h-screen bg-[#090e1a] flex items-center justify-center p-8">
        <div className="max-w-md w-full bg-red-500/10 border border-red-500/20 rounded-3xl p-8 text-center space-y-6">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto" />
          <h2 className="text-xl font-bold text-white">Connection Error</h2>
          <p className="text-slate-400 text-sm">{errorMsg}</p>
          <button 
            onClick={() => window.location.reload()}
            className="w-full bg-white/5 hover:bg-white/10 text-white font-bold py-3 rounded-2xl border border-white/10 transition-all"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#090e1a] text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 h-16 bg-[#0f172a]/80 backdrop-blur-xl border-b border-white/5 z-[100] flex items-center justify-between px-8">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-600/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-white leading-tight">Federated Learning Dashboard</h1>
            <p className="text-[10px] text-slate-500 font-medium uppercase tracking-widest">Cystic Fibrosis Diagnosis Model</p>
          </div>
        </div>

        <div className="flex items-center bg-black/20 p-1 rounded-xl border border-white/5">
          <NavBtn active={view === 'dashboard'} onClick={() => setView('dashboard')} icon={<LayoutDashboard className="w-4 h-4" />} label="Dashboard" />
          <NavBtn active={view === 'diagnose'} onClick={() => setView('diagnose')} icon={<Stethoscope className="w-4 h-4" />} label="Diagnose" />
          <NavBtn active={view === 'simulator'} onClick={() => setView('simulator')} icon={<Cpu className="w-4 h-4" />} label="Simulator" />
          <NavBtn active={false} onClick={() => {}} icon={<Shield className="w-4 h-4" />} label="Privacy" />
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-[10px] font-bold text-emerald-400">
            <span className="w-1.5 h-1.2 bg-emerald-400 rounded-full animate-pulse" />
            Model Ready
          </div>
        </div>
      </nav>

      <main className="pt-24 pb-12 px-8 max-w-[1600px] mx-auto">
        {view === 'dashboard' && dashboardData && <DashboardView data={dashboardData} />}
        {view === 'diagnose' && metadata && <DiagnoseView metadata={metadata} dashboardData={dashboardData} />}
        {view === 'simulator' && <SimulatorPlaceholder />}
      </main>
    </div>
  );
}

// --- Views ---

function DashboardView({ data }: { data: DashboardData }) {
  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Top Row: Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Final Accuracy" value={`${(data.summary.final_accuracy * 100).toFixed(2)}%`} trend="+12.4%" icon={<Monitor className="text-blue-400" />} />
        <StatCard title="F1-Score" value={data.summary.f1_score.toFixed(3)} trend="Global Average" icon={<TrendingUp className="text-indigo-400" />} />
        <StatCard title="Hospital Clients" value={data.summary.num_clients.toString()} trend={`${data.summary.total_samples?.toLocaleString()} samples`} icon={<User className="text-slate-400" />} />
        <StatCard title="Privacy Budget (ε)" value={data.summary.privacy_budget.toFixed(1)} trend="Strong privacy" icon={<Shield className="text-emerald-400" />} />
      </div>

      {/* Middle Row: Radar & Privacy Settings */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[400px]">
        <div className="lg:col-span-7 bg-[#0f172a] rounded-3xl border border-white/5 p-6 flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Metric Comparison across sites</h3>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={data.radar_metrics}>
                <PolarGrid stroke="#ffffff10" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Radar name="Hospital 4 (CF Clinic)" dataKey="h4" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                <Radar name="Hospital 3" dataKey="h3" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
                <Radar name="Hospital 2" dataKey="h2" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 10, paddingTop: 20 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="lg:col-span-5 bg-[#0f172a] rounded-3xl border border-white/5 p-6 space-y-4 overflow-y-auto">
          <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">Privacy & Security</h3>
          {data.privacy_settings.map((s, i) => (
            <PrivacyItem key={i} label={s.label} value={s.value} desc={s.desc} />
          ))}
        </div>
      </div>

      {/* Bottom Row: Progress & Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#0f172a] rounded-3xl border border-white/5 p-6">
          <div className="mb-6">
            <h3 className="text-lg font-bold text-white mb-1">Training Progress</h3>
            <p className="text-xs text-slate-500">Global model metrics over 10 FL rounds</p>
          </div>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.training_progress}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis dataKey="round" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '12px', fontSize: 12 }} 
                  itemStyle={{ padding: '2px 0' }}
                />
                <Legend iconType="circle" />
                <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 4, fill: '#8b5cf6' }} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="loss" name="Loss" stroke="#ef4444" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="f1" name="F1-Score" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-[#0f172a] rounded-3xl border border-white/5 p-6">
          <div className="mb-6">
            <h3 className="text-lg font-bold text-white mb-1">Non-IID Data Distribution</h3>
            <p className="text-xs text-slate-500">Sample counts and CF prevalence across 5 hospitals</p>
          </div>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.data_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis dataKey="hospital" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip 
                  cursor={{ fill: '#ffffff05' }}
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '12px', fontSize: 12 }} 
                />
                <Bar dataKey="samples" radius={[6, 6, 0, 0]}>
                  {data.data_distribution.map((_, index) => (
                    <Cell key={index} fill={getColorForHospital(index)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-around mt-4">
            {data.data_distribution.map((d, i) => (
              <div key={i} className="text-center">
                <div className="w-2 h-2 rounded-full mx-auto mb-1" style={{ backgroundColor: getColorForHospital(i) }} />
                <p className="text-[10px] font-bold text-slate-400">{(d.prevalence * 100).toFixed(1)}% CF</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Summary & Confusion Matrix */}
      <div className="bg-[#0f172a] rounded-3xl border border-white/5 p-8">
        <div className="mb-8">
          <h3 className="text-lg font-bold text-white mb-1">Model Performance Summary</h3>
          <p className="text-xs text-slate-500">Final evaluation metrics on global test set (n={(data.confusion_matrix.tp + data.confusion_matrix.fp + data.confusion_matrix.fn + data.confusion_matrix.tn).toLocaleString()})</p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Confusion Matrix */}
          <div className="lg:col-span-5">
            <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-4">Confusion Matrix</h4>
            <div className="grid grid-cols-2 gap-2">
              <MatrixCell label="True Positive" value={data.confusion_matrix.tp} color="bg-emerald-500/10 text-emerald-400 border-emerald-500/20" />
              <MatrixCell label="False Positive" value={data.confusion_matrix.fp} color="bg-red-500/10 text-red-400 border-red-500/20" />
              <MatrixCell label="False Negative" value={data.confusion_matrix.fn} color="bg-red-500/10 text-red-400 border-red-500/20" />
              <MatrixCell label="True Negative" value={data.confusion_matrix.tn} color="bg-emerald-500/10 text-emerald-400 border-emerald-500/20" />
            </div>
            <p className="text-[10px] text-center mt-4 text-slate-600 italic">Predicted: Columns | Actual: Rows</p>
          </div>

          {/* Detailed Metrics */}
          <div className="lg:col-span-7 grid grid-cols-2 md:grid-cols-3 gap-x-12 gap-y-6">
            <DetailedMetric label="Accuracy" value={`${(data.summary.final_accuracy * 100).toFixed(2)}%`} desc="Overall correct predictions" />
            <DetailedMetric label="Precision" value={`${((data.summary.precision || 0) * 100).toFixed(2)}%`} desc="True positives / Predicted positives" />
            <DetailedMetric label="Recall" value={`${((data.summary.recall || 0) * 100).toFixed(2)}%`} desc="True positives / Actual positives" />
            <DetailedMetric label="Specificity" value={`${((data.summary.specificity || 0) * 100).toFixed(2)}%`} desc="True negatives / Actual negatives" />
            <DetailedMetric label="AUC-ROC" value={(data.summary.auc || 0).toFixed(4)} desc="Area under ROC curve" />
            <DetailedMetric label="F1-Score" value={data.summary.f1_score.toFixed(4)} desc="Harmonic mean of precision & recall" />
          </div>
        </div>
      </div>
    </div>
  );
}

function DiagnoseView({ metadata, dashboardData }: { metadata: Metadata, dashboardData: DashboardData | null }) {
  const [activeTab, setActiveTab] = useState(metadata.schema[0].group);
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [mutationQuery, setMutationQuery] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const initial: Record<string, any> = {};
    metadata.schema.forEach((g) => {
      g.fields.forEach((f) => {
        if (f.type === "select") initial[f.name] = f.options?.[0];
        else initial[f.name] = 0;
      });
    });
    setFormData(initial);
  }, [metadata]);

  const handleInputChange = (name: string, value: any) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAnalyzing(true);
    setErrorMsg(null);
    try {
      const resp = await axios.post(`${API_BASE}/api/predict`, formData);
      setResult(resp.data);
    } catch (err: any) {
      setErrorMsg("Analysis failed. Please check inputs.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="mb-12 text-left">
        <h2 className="text-3xl font-black text-white mb-2 tracking-tight">CF Diagnostic Tool</h2>
        <p className="text-slate-500 max-w-2xl">Enter patient clinical data to assess Cystic Fibrosis risk using the federated learning model</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Form Column */}
        <div className="lg:col-span-7 bg-[#0f172a] rounded-3xl border border-white/5 overflow-hidden shadow-2xl">
          <div className="p-8 border-b border-white/5 bg-white/2 flex items-center gap-4">
            <div className="bg-blue-500/20 p-2 rounded-xl text-blue-400">
              <Stethoscope className="w-6 h-6" />
            </div>
            <div>
              <h3 className="font-bold text-white tracking-tight leading-none mb-1">Patient Clinical Data Entry</h3>
              <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">Enter all available clinical information for accurate risk assessment</p>
            </div>
          </div>

          <div className="p-2 bg-black/20 border-b border-white/5 flex gap-1 overflow-x-auto no-scrollbar">
            {metadata.schema.map(g => (
              <button 
                key={g.group}
                onClick={() => setActiveTab(g.group)}
                className={cn(
                  "px-6 py-2.5 rounded-xl text-[11px] font-bold transition-all whitespace-nowrap",
                  activeTab === g.group ? "bg-white/10 text-white shadow-lg" : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
                )}
              >
                {g.group.split(' ')[0]}
              </button>
            ))}
            <button 
              onClick={() => setActiveTab('Diagnostic')}
              className={cn(
                "px-6 py-2.5 rounded-xl text-[11px] font-bold transition-all",
                activeTab === 'Diagnostic' ? "bg-white/10 text-white" : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
              )}
            >
              Diagnostic
            </button>
          </div>

          <form onSubmit={handleSubmit} className="p-8 space-y-8">
            {errorMsg && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 text-red-500 text-sm font-bold rounded-2xl animate-in slide-in-from-top-2 duration-300">
                {errorMsg}
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 min-h-[300px]">
              {activeTab === 'Diagnostic' ? (
                <div className="col-span-2 space-y-6">
                  <div className="bg-blue-500/5 border border-blue-500/10 rounded-2xl p-6">
                    <div className="flex items-center gap-3 mb-4 text-blue-400">
                      <Dna className="w-5 h-5" />
                      <h4 className="font-bold">Genetic Confirmation</h4>
                    </div>
                    <div className="relative">
                      <input 
                        list="mutation-list"
                        placeholder="Search for CFTR mutations..."
                        className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                        value={mutationQuery}
                        onChange={(e) => {
                          setMutationQuery(e.target.value);
                          handleInputChange("mutation", e.target.value);
                        }}
                      />
                      <datalist id="mutation-list">
                        {metadata.mutations.map(m => <option key={m.name} value={m.name}>{m.determination}</option>)}
                      </datalist>
                    </div>
                  </div>
                </div>
              ) : (
                metadata.schema.find(g => g.group === activeTab)?.fields.map(field => (
                  <div key={field.name} className="space-y-2">
                    <label className="text-[11px] font-bold text-slate-500 uppercase tracking-widest ml-1">{field.label}</label>
                    {field.type === "select" ? (
                      <select 
                        className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none appearance-none"
                        value={formData[field.name]}
                        onChange={(e) => handleInputChange(field.name, e.target.value)}
                      >
                        {field.options?.map(o => <option key={o} value={o}>{o === 0 ? 'No' : o === 1 ? 'Yes' : o}</option>)}
                      </select>
                    ) : (
                      <input 
                        type="number"
                        className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-sm text-white focus:ring-2 focus:ring-blue-500 outline-none"
                        value={formData[field.name]}
                        onChange={(e) => handleInputChange(field.name, e.target.value)}
                      />
                    )}
                  </div>
                ))
              )}
            </div>

            <div className="flex gap-4 pt-4 border-t border-white/5">
              <button 
                type="submit"
                disabled={analyzing}
                className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold py-4 rounded-2xl flex items-center justify-center gap-3 transition-all active:scale-[0.98] shadow-xl shadow-blue-600/20"
              >
                {analyzing ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Activity className="w-5 h-5" />}
                Run Diagnosis
              </button>
              <button 
                type="button"
                onClick={() => setFormData({})}
                className="px-8 bg-white/5 hover:bg-white/10 text-slate-300 font-bold rounded-2xl transition-all border border-white/5"
              >
                Reset Form
              </button>
            </div>
          </form>
        </div>

        {/* Results Column */}
        <div className="lg:col-span-5 bg-[#0f172a] rounded-3xl border border-white/5 p-8 h-full min-h-[600px] flex flex-col items-center justify-center relative">
          <div className="absolute top-8 left-8 flex items-center gap-2 text-blue-400">
            <Activity className="w-5 h-5" />
            <h3 className="font-bold tracking-tight text-white uppercase text-[10px] tracking-widest">Diagnosis Result</h3>
          </div>

          {result ? (
            <div className="w-full text-center space-y-8 animate-in zoom-in-95 duration-500">
              <div className="text-8xl font-black tracking-tighter text-blue-500 drop-shadow-[0_0_30px_rgba(59,130,246,0.3)]">
                {(result.probability * 100).toFixed(1)}%
              </div>
              <p className="text-slate-400 font-medium tracking-wide">Probability of Cystic Fibrosis</p>
              
              <div className={cn(
                "px-8 py-3 rounded-full text-white font-black text-xl shadow-2xl border-4 border-white/10 mx-auto inline-block",
                result.risk_level === 'High' ? 'bg-red-600' : result.risk_level === 'Moderate' ? 'bg-amber-600' : 'bg-emerald-600'
              )}>
                {result.risk_level} Risk
              </div>

              <div className="bg-white/2 rounded-2xl p-6 border border-white/5 text-left w-full space-y-4">
                <div className="flex justify-between items-center text-sm border-b border-white/5 pb-3">
                  <span className="text-slate-500">Neural Confidence</span>
                  <span className="text-white font-bold">{dashboardData ? (dashboardData.summary.final_accuracy * 100).toFixed(1) : '...'}% Acc</span>
                </div>
                <div className="flex justify-between items-center text-sm border-b border-white/5 pb-3">
                  <span className="text-slate-500">Global F1-Score</span>
                  <span className="text-indigo-400 font-bold">{dashboardData ? dashboardData.summary.f1_score.toFixed(3) : '...'}</span>
                </div>
                {formData.mutation && (
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">Genetic Modifier</span>
                    <span className="text-blue-400 font-bold">{formData.mutation}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-6 opacity-20 group">
              <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center border-4 border-slate-700/50 transition-transform group-hover:scale-110 duration-500">
                <AlertCircle className="w-10 h-10 text-slate-400" />
              </div>
              <p className="text-center text-slate-400 max-w-[200px] font-medium leading-relaxed">Enter patient clinical data and click "Run Diagnosis" to get a CF risk assessment</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Helpers & Small Components ---

function LoadingScreen() {
  return (
    <div className="min-h-screen bg-[#090e1a] flex items-center justify-center">
      <div className="flex flex-col items-center gap-6">
        <div className="relative">
          <Activity className="w-12 h-12 text-blue-600 animate-pulse" />
          <div className="absolute inset-0 bg-blue-600/20 blur-2xl rounded-full" />
        </div>
        <p className="text-slate-500 font-black uppercase tracking-[0.3em] animate-pulse">CFVision System Loading</p>
      </div>
    </div>
  );
}

function NavBtn({ active, icon, label, onClick }: { active: boolean, icon: any, label: string, onClick: () => void }) {
  return (
    <button 
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-5 py-2.5 rounded-xl text-[11px] font-bold transition-all",
        active ? "bg-white/10 text-white shadow-lg border border-white/5" : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function StatCard({ title, value, trend, icon }: { title: string, value: string, trend: string, icon: any }) {
  return (
    <div className="bg-[#0f172a] p-6 rounded-3xl border border-white/5 space-y-4 hover:border-white/10 transition-colors group">
      <div className="flex items-center justify-between">
        <div className="bg-white/2 p-2.5 rounded-2xl group-hover:scale-110 transition-transform">{icon}</div>
        <div className="w-8 h-8 bg-blue-500/5 rounded-full flex items-center justify-center text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">
          <ArrowUpRight className="w-4 h-4" />
        </div>
      </div>
      <div>
        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">{title}</p>
        <div className="flex items-baseline gap-3">
          <h4 className="text-2xl font-black text-white tracking-tight">{value}</h4>
          <span className={cn(
            "text-[10px] font-bold",
            trend.includes('+') ? "text-emerald-400" : trend.includes('Strong') ? "text-emerald-400" : "text-slate-500"
          )}>{trend}</span>
        </div>
      </div>
    </div>
  );
}

function PrivacyItem({ label, value, desc }: { label: string, value: string, desc: string }) {
  return (
    <div className="flex items-start gap-4 p-4 rounded-2xl bg-white/2 border border-white/5 hover:bg-white/5 transition-colors">
      <div className="bg-blue-500/10 p-2.5 rounded-xl text-blue-400">
        {label.includes('Budget') ? <Shield className="w-5 h-5" /> : label.includes('Clipping') ? <Monitor className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
      </div>
      <div className="flex-1">
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs font-bold text-white">{label}</span>
          <span className="text-xs font-black text-blue-400">{value}</span>
        </div>
        <p className="text-[10px] text-slate-500 font-medium leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}

function MatrixCell({ label, value, color }: { label: string, value: number, color: string }) {
  return (
    <div className={cn("p-4 rounded-2xl border flex flex-col items-center justify-center text-center", color)}>
      <span className="text-2xl font-black tracking-tight">{value}</span>
      <span className="text-[9px] font-bold uppercase tracking-widest opacity-60 mt-1">{label}</span>
    </div>
  );
}

function DetailedMetric({ label, value, desc }: { label: string, value: string, desc: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-baseline border-b border-white/5 pb-1">
        <span className="text-[11px] font-bold text-slate-500">{label}</span>
        <span className="text-sm font-black text-blue-400 tracking-tight">{value}</span>
      </div>
      <p className="text-[9px] text-slate-600 leading-tight">{desc}</p>
    </div>
  );
}

function SimulatorPlaceholder() {
  return (
    <div className="h-[600px] flex flex-col items-center justify-center text-center space-y-6 opacity-50">
      <Cpu className="w-16 h-16 text-blue-500 animate-pulse" />
      <h2 className="text-2xl font-black">Edge Device Simulator</h2>
      <p className="text-slate-500 max-w-sm">Configure hardware constraints and network conditions to observe federated training impact in real-time.</p>
      <div className="px-6 py-2 bg-white/5 border border-white/10 rounded-full text-xs font-bold uppercase tracking-widest text-slate-400 italic">Module under integration</div>
    </div>
  );
}

function getColorForHospital(index: number) {
  const colors = ['#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4'];
  return colors[index % colors.length];
}
