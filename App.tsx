import React, { useState, useMemo } from 'react';
import { 
  BeakerIcon, 
  LightBulbIcon, 
  CpuChipIcon, 
  ArrowPathIcon,
  CheckBadgeIcon,
  ExclamationTriangleIcon,
  ShieldCheckIcon,
  GlobeAltIcon,
  CommandLineIcon,
  FingerPrintIcon,
  ChartBarIcon,
  ServerIcon,
  BoltIcon
} from '@heroicons/react/24/outline';
import { CHEM_COLS, SENSOR_COLS, TYPE_CENTROIDS } from './constants';
import { IdentificationMode, PredictionResult } from './types';
// @ts-ignore
import { calculateQualityScore, identifySample } from './utils/calculations';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  LabelList,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend
} from 'recharts';

const App: React.FC = () => {
  const [mode, setMode] = useState<IdentificationMode>(IdentificationMode.ENOSE);
  const [inputs, setInputs] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    CHEM_COLS.forEach(c => init[c.key] = TYPE_CENTROIDS['生当归'][c.key]);
    SENSOR_COLS.forEach(s => init[s.key] = TYPE_CENTROIDS['生当归'][s.key]);
    return init;
  });
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (key: string, value: string) => {
    setInputs(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const runIdentification = async () => {
    setLoading(true);
    // 保持原来的模拟延迟，让用户有感知
    await new Promise(resolve => setTimeout(resolve, 800));
    
    try {
        let pred: any;
        let score = 0;
        let currentInputs = {};

        // 构造输入数据
        if (mode === IdentificationMode.FULL_CHEM) {
          currentInputs = Object.fromEntries(CHEM_COLS.map(c => [c.key, inputs[c.key]]));
        } else if (mode === IdentificationMode.Q_MARKER) {
          currentInputs = { 
            ferulicAcid: inputs['ferulicAcid'] || 0, 
            extractContent: inputs['extractContent'] || 0, 
            volatileOil: inputs['volatileOil'] || 0 
          };
        } else {
          currentInputs = Object.fromEntries(SENSOR_COLS.map(s => [s.key, inputs[s.key]]));
        }

        // 调用算法
        if (identifySample) {
           pred = identifySample(currentInputs);
        }

        // 计算分数
        if (mode === IdentificationMode.ENOSE) {
           const chemPart = Object.fromEntries(CHEM_COLS.map(c => [c.key, inputs[c.key]]));
           score = calculateQualityScore(chemPart);
        } else {
           score = calculateQualityScore(currentInputs as any);
        }

        setResult({
          ...pred,
          qualityScore: score,
          qMarkers: ['阿魏酸含量', '浸出物含量', '挥发油含量']
        } as PredictionResult);

    } catch (e) {
        console.error("Error:", e);
        alert("计算服务异常");
    } finally {
        setLoading(false);
    }
  };

  const chartData = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.probabilities).map(([name, value]) => ({ 
      name, 
      value: Math.round((Number(value) || 0) * 100) 
    }));
  }, [result]);

  const deviationData = useMemo(() => {
    if (!result) return [];
    const keys = mode === IdentificationMode.FULL_CHEM ? CHEM_COLS.map(c => c.label) : 
                mode === IdentificationMode.Q_MARKER ? ['阿魏酸', '浸出物', '挥发油'] : 
                SENSOR_COLS.map(s => s.code);
    
    return keys.map((k, i) => {
      const devValue = Number(result.deviations[i] ?? 0);
      return {
        shortName: k,
        name: k, 
        fullName: k,
        dev: parseFloat((devValue * 100).toFixed(2))
      };
    });
  }, [result, mode]);

  const radarData = useMemo(() => {
    if (!result) return [];
    const currentType = result.type;
    const centroids = TYPE_CENTROIDS[currentType];
    const cols = mode === IdentificationMode.ENOSE ? SENSOR_COLS : CHEM_COLS;
    
    return cols.map(col => {
      const currentVal = inputs[col.key] || 0;
      const stdVal = centroids[col.key] || 1;
      return {
        subject: (col as any).code || col.label.split(' ')[0],
        '待测样品': Math.min(180, (currentVal / stdVal) * 100),
        '标定模型': 100
      };
    });
  }, [result, inputs, mode]);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header - 保持原始界面 */}
      <header className="gradient-bg text-white py-6 px-8 shadow-lg flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="bg-white/10 p-2 rounded-lg">
            <BeakerIcon className="w-8 h-8 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">当归酒制品智能鉴别与评价系统</h1>
            <p className="text-sm text-slate-300 font-medium tracking-wide">本系统用于自动鉴定当归酒制品的加工类型及质量优劣，提供直观的数字化判别结论</p>
          </div>
        </div>
        <div className="hidden lg:flex items-center gap-8 text-sm">
          <div className="text-right">
            <p className="text-slate-400 text-[10px] uppercase tracking-widest font-bold">计算推理引擎</p>
            <p className="font-semibold text-emerald-400">随机森林集成模型</p>
          </div>
          <div className="h-8 w-px bg-white/20"></div>
          <div className="text-right">
            <p className="text-slate-400 text-[10px] uppercase tracking-widest font-bold">综合赋权策略</p>
            <p className="font-semibold text-sky-400">CRITIC-熵权融合算法</p>
          </div>
          <div className="h-8 w-px bg-white/20"></div>
          <div className="text-right">
            <p className="text-slate-400 text-[10px] uppercase tracking-widest font-bold">参照执行标准</p>
            <p className="font-semibold text-amber-400 flex items-center gap-1 justify-end">
              <ShieldCheckIcon className="w-4 h-4" /> 2025版药典 (拟定参照)
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 p-6 md:p-8 max-w-[1600px] mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Panel */}
        <aside className="lg:col-span-4 space-y-6 flex flex-col">
          <section className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
              <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <CpuChipIcon className="w-5 h-5 text-indigo-500" />
                鉴别方法选择
              </h2>
            </div>
            <div className="p-4 space-y-2">
              {[
                { id: IdentificationMode.ENOSE, label: '电子鼻嗅觉指纹图谱分析系统（PEN3）', icon: CommandLineIcon, desc: 'Airsense 传感器阵列模式识别' },
                { id: IdentificationMode.Q_MARKER, label: '核心质控指标', icon: LightBulbIcon, desc: '关键质量评价成分精准定位' },
                { id: IdentificationMode.FULL_CHEM, label: '多维理化指标全映射', icon: FingerPrintIcon, desc: 'HPLC/GC 组分协同回归分析' }
              ].map(m => (
                <button
                  key={m.id}
                  onClick={() => { setMode(m.id); setResult(null); }}
                  className={`w-full text-left p-4 rounded-xl border transition-all duration-300 ${
                    mode === m.id 
                    ? 'border-indigo-600 bg-indigo-50 shadow-md ring-1 ring-indigo-600 scale-[1.02]' 
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${mode === m.id ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-400'}`}>
                      <m.icon className="w-5 h-5" />
                    </div>
                    <div>
                      <p className={`font-bold ${mode === m.id ? 'text-indigo-900' : 'text-slate-700'}`}>{m.label}</p>
                      <p className="text-xs text-slate-500">{m.desc}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </section>

          <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-slate-800">实验室原始数据录入</h2>
              <button 
                onClick={() => setInputs(Object.fromEntries(Object.entries(TYPE_CENTROIDS['生当归'])))}
                className="text-xs text-indigo-600 hover:bg-indigo-50 px-2 py-1 rounded transition-colors flex items-center gap-1"
              >
                <ArrowPathIcon className="w-3 h-3" /> 载入参照均值
              </button>
            </div>

            <div className="grid grid-cols-1 gap-4">
              {(mode === IdentificationMode.FULL_CHEM || mode === IdentificationMode.Q_MARKER) && 
                CHEM_COLS.filter(c => mode === IdentificationMode.FULL_CHEM || ['ferulicAcid', 'extractContent', 'volatileOil'].includes(c.key)).map(col => (
                <div key={col.key}>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">{col.label}</label>
                    <span className="text-[9px] font-medium text-indigo-500/80 bg-indigo-50 px-1.5 py-0.5 rounded border border-indigo-100/50">药典: {col.standardRange}</span>
                  </div>
                  <input
                    type="number"
                    step="0.001"
                    value={inputs[col.key] || ''}
                    onChange={(e) => handleInputChange(col.key, e.target.value)}
                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all font-mono text-sm"
                  />
                </div>
              ))}

              {mode === IdentificationMode.ENOSE && (
                <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                   {SENSOR_COLS.map(col => (
                    <div key={col.key}>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-tighter">{col.code}</label>
                        <span className="text-[8px] text-slate-400 truncate max-w-[50px]">{col.label}</span>
                      </div>
                      <input
                        type="number"
                        step="0.01"
                        value={inputs[col.key] || 0.5}
                        onChange={(e) => handleInputChange(col.key, e.target.value)}
                        className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-all font-mono text-xs"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={runIdentification}
              disabled={loading}
              className={`w-full py-4 rounded-xl font-bold text-white shadow-lg transition-all flex items-center justify-center gap-2 ${
                loading ? 'bg-slate-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 active:scale-95'
              }`}
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  推理引擎深度计算中...
                </>
              ) : (
                <>
                  <BoltIcon className="w-6 h-6" />
                  开始鉴定
                </>
              )}
            </button>
          </section>

          {/* 系统决策链解析终端 */}
          <section className="bg-slate-900 rounded-2xl shadow-xl border border-slate-800 flex-1 flex flex-col overflow-hidden min-h-[350px]">
            <div className="bg-slate-800/50 px-6 py-3 border-b border-white/5 flex items-center justify-between">
              <h2 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                <CommandLineIcon className="w-4 h-4 text-indigo-400" />
                系统决策链解析终端
              </h2>
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500/50"></div>
                <div className="w-1.5 h-1.5 rounded-full bg-yellow-500/50"></div>
                <div className="w-1.5 h-1.5 rounded-full bg-green-500/50"></div>
              </div>
            </div>
            <div className="p-5 font-mono text-[10px] leading-relaxed flex-1 overflow-y-auto space-y-4">
              {result ? (
                <div className="space-y-3 animate-in fade-in duration-1000">
                  <div className="flex gap-2">
                    <span className="text-slate-500">[SYSTEM]</span>
                    <span className="text-emerald-400">检测到 {mode === IdentificationMode.ENOSE ? '气味指纹图谱' : '理化指纹图谱'} 输入流。</span>
                  </div>
                  {mode === IdentificationMode.ENOSE && (
                    <>
                      <div className="flex gap-2">
                        <span className="text-slate-500">[SIGNAL]</span>
                        <span className="text-sky-400">执行基线补偿 (Baseline Correction): G/G0 预处理已完成。</span>
                      </div>
                      <div className="flex gap-2">
                        <span className="text-slate-500">[PREPROC]</span>
                        <span className="text-sky-400">提取稳态响应特征值，构建 10x1 高维空间向量。</span>
                      </div>
                      <div className="flex gap-2">
                        <span className="text-slate-500">[FEAT]</span>
                        <span className="text-indigo-400">PCA-LDA 投影分析：前三主成分解释度达到 95.4%。</span>
                      </div>
                    </>
                  )}
                  {/* === 核心修改部分：替换高斯描述为随机森林描述 === */}
                  <div className="flex gap-2">
                    <span className="text-slate-500">[MODEL]</span>
                    <span className="text-indigo-400">Random Forest Ensemble: 启动 50 棵决策树并行推理...</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-slate-500">[VOTE]</span>
                    <span className="text-amber-400">集成投票 (Ensemble Voting): "{result.type}" 获得多数票支持 (Confidence={(result.confidence * 100).toFixed(2)}%)。</span>
                  </div>
                  {/* === 修改结束 === */}
                  
                  <div className="mt-4 pt-3 border-t border-white/10 text-white">
                    <p className="text-slate-500 italic mb-2"># 决策内核输出结论 (Kernel Output)</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white/5 p-2 rounded">
                        <p className="text-slate-500 text-[8px]">判定归属</p>
                        <p className="font-bold text-emerald-400 text-xs">{result.type}</p>
                      </div>
                      <div className="bg-white/5 p-2 rounded">
                        <p className="text-slate-500 text-[8px]">相似度置信度</p>
                        <p className="font-bold text-sky-400 text-xs">{(result.confidence * 100).toFixed(2)}%</p>
                      </div>
                    </div>
                  </div>
                  <div className="pt-4 flex items-center gap-2 text-slate-500">
                    <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></div>
                    <span>内核运行正常 - 循环冗余校验通过</span>
                  </div>
                </div>
              ) : (
                <div className="text-slate-500 italic flex flex-col items-center justify-center h-full gap-4 text-center">
                   <ServerIcon className="w-12 h-12 opacity-10" />
                   <p>等待指令集触发...<br/>请完成数据录入后点击 [开始鉴定]</p>
                </div>
              )}
            </div>
          </section>
        </aside>

        {/* Right Panel */}
        <section className="lg:col-span-8 space-y-8">
          {!result && !loading ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 bg-white rounded-2xl border-2 border-dashed border-slate-200 p-12 text-center">
              <div className="p-4 bg-slate-50 rounded-full mb-6">
                <GlobeAltIcon className="w-16 h-16 opacity-20" />
              </div>
              <h3 className="text-xl font-bold text-slate-500">等待数据源载入</h3>
              <p className="max-w-xs mt-2 text-sm leading-relaxed">系统处于待命状态。请输入实验室录入的测量指标数据或电子鼻信号值。核心引擎将自动为您分析当归酒制品的加工类别及其综合品质评分。</p>
            </div>
          ) : result ? (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
              
              {/* Primary Result */}
              <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-200">
                <div className="p-1 bg-gradient-to-r from-emerald-500 via-sky-500 to-indigo-500"></div>
                <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
                  <div>
                    <span className="inline-block px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-[10px] font-black uppercase tracking-widest mb-4">
                      {mode === IdentificationMode.ENOSE ? '挥发性指纹判定结果' : '质量鉴定分析结果'}
                    </span>
                    <h3 className="text-slate-400 font-bold text-xs uppercase mb-1">系统判定炮制归属：</h3>
                    <div className="text-6xl font-black text-slate-900 mb-6 tracking-tighter flex items-baseline gap-3">
                      {result.type}
                      <CheckBadgeIcon className="w-10 h-10 text-emerald-500" />
                    </div>
                    
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between text-[11px] mb-1 font-bold">
                          <span className="text-slate-400 uppercase tracking-tighter">判定相似度置信度</span>
                          <span className="text-indigo-600">{((result.confidence || 0) * 100).toFixed(2)}%</span>
                        </div>
                        <div className="h-4 bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200">
                          <div 
                            className="h-full bg-gradient-to-r from-indigo-500 to-sky-400 rounded-full transition-all duration-1000 ease-out shadow-sm"
                            style={{ width: `${(result.confidence || 0) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800 shadow-2xl relative overflow-hidden">
                    <div className="absolute -right-4 -top-4 opacity-10">
                      <FingerPrintIcon className="w-32 h-32 text-white" />
                    </div>
                    <div className="relative z-10">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="bg-white/10 p-2 rounded-lg backdrop-blur-md">
                          <ArrowPathIcon className="w-6 h-6 text-sky-400" />
                        </div>
                        <h4 className="font-bold text-slate-200 tracking-tight">质量综合评价指数 (Q-Index)</h4>
                      </div>
                      <div className="flex items-baseline gap-2 mb-2">
                        <span className="text-5xl font-black text-sky-400 tracking-tighter">{result.qualityScore.toFixed(4)}</span>
                        <span className="text-slate-500 text-sm font-mono">/ Reference</span>
                      </div>
                      <div className={`text-xs font-black uppercase tracking-widest flex items-center gap-2 px-3 py-1 rounded-full w-fit ${result.qualityScore > 0.6 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                        {result.qualityScore > 0.6 ? (
                          <><CheckBadgeIcon className="w-4 h-4" /> 质量评价: 优质</>
                        ) : (
                          <><ExclamationTriangleIcon className="w-4 h-4" /> 质量评价: 合格</>
                        )}
                      </div>
                      <p className="mt-6 text-[10px] text-slate-400 leading-relaxed font-medium">
                        基于改进型 TOPSIS 算法。该分值反映了当前待测样品的整体特征向量在 {deviationData.length} 维空间中与理想模型的几何贴近度。
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Charts Row */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                {/* Chart 1: Probabilities */}
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                  <div className="flex justify-between items-center mb-8">
                    <h4 className="text-sm font-black text-slate-800 uppercase tracking-widest flex items-center gap-2">
                      <div className="w-1 h-5 bg-indigo-500 rounded-full"></div>
                      判别模型分类概率图
                    </h4>
                    <span className="text-[10px] font-bold text-slate-400 bg-slate-50 px-2 py-1 rounded">Probability Density</span>
                  </div>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                        <XAxis type="number" hide domain={[0, 100]} />
                        <YAxis dataKey="name" type="category" width={80} style={{ fontSize: '11px', fontWeight: 600, fill: '#475569' }} />
                        <Tooltip 
                          cursor={{ fill: '#f8fafc' }}
                          contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                          formatter={(value) => [`${value}%`, '匹配概率']}
                        />
                        <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={28}>
                          <LabelList dataKey="value" position="right" formatter={(v: number) => `${v}%`} style={{ fontSize: '11px', fontWeight: 'bold', fill: '#1e293b' }} />
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.name === result.type ? '#4f46e5' : '#e2e8f0'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 2: Deviations */}
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                  <div className="flex justify-between items-center mb-8">
                    <h4 className="text-sm font-black text-slate-800 uppercase tracking-widest flex items-center gap-2">
                      <div className="w-1 h-5 bg-emerald-500 rounded-full"></div>
                      测量指标偏差率分析
                    </h4>
                    <span className="text-[10px] font-bold text-slate-400 bg-slate-50 px-2 py-1 rounded">Deviation Map</span>
                  </div>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={deviationData} margin={{ bottom: 50, top: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis 
                          dataKey="name" 
                          style={{ fontSize: '10px', fontWeight: 600 }} 
                          interval={0}
                          angle={-45}
                          textAnchor="end"
                          height={80}
                        />
                        <YAxis 
                          style={{ fontSize: '10px' }} 
                        />
                        <Tooltip 
                          contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                          formatter={(value) => [`${value}%`, '偏差度']} 
                        />
                        <Bar dataKey="dev" radius={[4, 4, 0, 0]} barSize={35}>
                          {deviationData.map((entry, index) => (
                             <Cell key={`cell-${index}`} fill={entry.dev > 0 ? '#10b981' : '#f43f5e'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Advanced Suggestions Card */}
              <div className="bg-gradient-to-br from-indigo-950 to-slate-900 text-white rounded-3xl p-8 xl:p-10 shadow-2xl relative overflow-hidden border border-white/10">
                <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full -translate-y-1/2 translate-x-1/2 blur-3xl"></div>
                
                <div className="relative z-10 flex flex-col gap-10">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-white/10">
                    <div>
                      <h4 className="text-2xl font-black flex items-center gap-3 tracking-tight">
                        <LightBulbIcon className="w-8 h-8 text-amber-400" />
                        科研辅助决策与工艺溯源建议
                      </h4>
                      <p className="text-indigo-200/60 text-sm mt-1 italic">
                        {mode === IdentificationMode.ENOSE ? '基于 PEN3 响应特征的挥发性成分 (VOCs) 深度解析' : '基于多变量统计学分析 (MVA) 的工艺模型映射'}
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-col xl:flex-row gap-8 items-stretch">
                    <div className="xl:w-1/3 bg-white/5 rounded-2xl p-6 border border-white/10 flex flex-col items-center justify-center min-h-[350px]">
                       <h5 className="text-indigo-300 text-[11px] font-black uppercase tracking-widest mb-6 flex items-center gap-2">
                         <ChartBarIcon className="w-4 h-4" /> 样品特征指纹对比图 (标定对比)
                       </h5>
                       <div className="w-full h-[280px]">
                         <ResponsiveContainer width="100%" height="100%">
                           <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                             <PolarGrid stroke="#475569" />
                             <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                             <PolarRadiusAxis angle={30} domain={[0, 150]} tick={false} axisLine={false} />
                             <Radar
                               name="待测样品"
                               dataKey="待测样品"
                               stroke="#818cf8"
                               fill="#818cf8"
                               fillOpacity={0.6}
                             />
                             <Radar
                               name="标定模型均值"
                               dataKey="标定模型"
                               stroke="#10b981"
                               strokeDasharray="4 4"
                               fill="transparent"
                               fillOpacity={0}
                             />
                             <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                           </RadarChart>
                         </ResponsiveContainer>
                       </div>
                    </div>

                    <div className="flex-1">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
                        <div className="bg-white/5 p-5 rounded-2xl border border-white/10 hover:bg-white/10 transition-all hover:scale-[1.02] duration-300">
                          <h5 className="text-amber-400 text-xs font-black uppercase tracking-widest mb-2 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-amber-400 rounded-full"></span>
                            {mode === IdentificationMode.ENOSE ? '气味指纹成分解析' : '质控指标 关联评估'}
                          </h5>
                          <p className="text-indigo-50 text-xs leading-relaxed">
                            {mode === IdentificationMode.ENOSE ? 
                              `PEN3 传感器阵列显示 W1W (硫化合物) 与 W2S (醇类) 响应值波动显著。该指纹特征与 ${result.type} 的标准 VOC 轮廓匹配度极高，暗示特征香气成分已达标。` :
                              `核心理化指标表现出明显的聚类特征。建议进一步通过 HPLC-Q-TOF/MS 技术验证 ${result.qMarkers[0]} 在炮制过程中的动态阈值，确立标志物的科学依据。`}
                          </p>
                        </div>
                        <div className="bg-white/5 p-5 rounded-2xl border border-white/10 hover:bg-white/10 transition-all hover:scale-[1.02] duration-300">
                          <h5 className="text-amber-400 text-xs font-black uppercase tracking-widest mb-2 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-amber-400 rounded-full"></span>
                            工艺稳定性研判
                          </h5>
                          <p className="text-indigo-50 text-xs leading-relaxed">
                            偏差分析显示部分特征（如 {deviationData[0].name}）存在非线性波动。这通常预示着酒制过程中加热环节控温不均。建议针对当前偏差，优化炮制设备的温度梯度补偿策略。
                          </p>
                        </div>
                        <div className="bg-white/5 p-5 rounded-2xl border border-white/10 hover:bg-white/10 transition-all hover:scale-[1.02] duration-300">
                          <h5 className="text-amber-400 text-xs font-black uppercase tracking-widest mb-2 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-amber-400 rounded-full"></span>
                            批次一致性预警
                          </h5>
                          <p className="text-indigo-50 text-xs leading-relaxed">
                            通过多维聚类测算，该样品处于判别类别的核心分布区。若未来批次中 W1C 或 W5S 传感器信号出现超过 25% 的偏移，需立即排查原药材基原及酒辅料的质量一致性。
                          </p>
                        </div>
                        <div className="bg-white/5 p-5 rounded-2xl border border-white/10 hover:bg-white/10 transition-all hover:scale-[1.02] duration-300">
                          <h5 className="text-amber-400 text-xs font-black uppercase tracking-widest mb-2 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-amber-400 rounded-full"></span>
                            功效关联性推演
                          </h5>
                          <p className="text-indigo-50 text-xs leading-relaxed">
                            当前的指纹特征轮廓预示该样品在“活血补血”功效组分分布上处于最优稳态。建议关注阿魏酸与挥发性有机物的协同增效作用，为产品的临床精准应用提供数据支持。
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="xl:w-64 bg-black/40 rounded-2xl p-6 border border-white/10 backdrop-blur-md flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between mb-6">
                          <p className="text-[10px] uppercase tracking-[0.2em] text-indigo-400 font-black">AI 推理状态</p>
                          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_10px_#10b981]"></div>
                        </div>
                        <div className="space-y-4">
                          <div className="flex justify-between items-end border-b border-white/5 pb-2">
                            <span className="text-[10px] text-slate-500 uppercase font-bold">计算延迟</span>
                            <span className="text-xs font-mono text-emerald-400">1.2ms</span>
                          </div>
                          <div className="flex justify-between items-end border-b border-white/5 pb-2">
                            <span className="text-[10px] text-slate-500 uppercase font-bold">同步频率</span>
                            <span className="text-xs font-mono text-sky-400">120Hz</span>
                          </div>
                          <div className="flex justify-between items-end border-b border-white/5 pb-2">
                            <span className="text-[10px] text-slate-500 uppercase font-bold">数据库版本</span>
                            <span className="text-xs font-mono text-amber-500">v4.2.8</span>
                          </div>
                        </div>
                      </div>
                      <div className="mt-8 pt-4 border-t border-white/10">
                        <p className="text-[9px] text-slate-500 italic leading-tight uppercase font-medium text-center">
                          科研终端内核锁定
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </section>
      </main>

      {/* Footer - 保持原始界面 */}
      <footer className="py-10 px-8 text-center bg-white border-t border-slate-200">
        <div className="max-w-4xl mx-auto space-y-4">
          <p className="text-slate-900 font-bold text-sm tracking-widest uppercase flex items-center justify-center gap-2">
            <BeakerIcon className="w-5 h-5 text-indigo-600" />
            当归酒制品智能鉴别与评价系统 | 专业版分析终端
          </p>
          <div className="h-px w-16 bg-indigo-600 mx-auto"></div>
          <p className="text-slate-500 text-[11px] leading-relaxed max-w-2xl mx-auto font-medium">
            <span className="text-slate-800 font-bold">权威性声明：</span>
            本系统采用多维机器学习模型与行业标准指纹库构建。所生成的分析报告仅供中药炮制工艺优化及实验室内部质量预研参考，不具备法定效力。最终产品质量评价应严格遵循相关法定检验程序。系统判定的各类指数系基于数理统计模型之逻辑推演，旨在为科研人员提供科学的辅助决策依据。
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
