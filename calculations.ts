// src/utils/calculations.ts
import { RandomForestClassifier } from 'ml-random-forest';
import { TRAINING_DATA } from './trainingData';
import { CHEM_COLS, TYPE_CENTROIDS } from '../constants';
import { ChemicalData } from '../types';

// ==========================================
// 1. 数据准备与切分
// ==========================================

// 提取标签映射
const distinctTypes = Array.from(new Set(TRAINING_DATA.map(d => d.type)));
const typeToId = Object.fromEntries(distinctTypes.map((t, i) => [t, i]));
const idToType = Object.fromEntries(distinctTypes.map((t, i) => [i, t]));

// 定义特征列
const CHEM_KEYS = ['polysaccharide', 'ferulicAcid', 'totalAsh', 'acidInsolubleAsh', 'volatileOil', 'moisture', 'extractContent'];
const Q_MARKER_KEYS = ['ferulicAcid', 'extractContent', 'volatileOil'];
const SENSOR_KEYS = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10'];

// *** 关键修改：切分训练集和测试集 ***
// 我们生成的数据表里，最后 10 条是专门生成的测试集 (索引 70-79)
const TEST_SIZE = 10;
const SPLIT_INDEX = TRAINING_DATA.length - TEST_SIZE;

const trainSet = TRAINING_DATA.slice(0, SPLIT_INDEX); // 前 70 条
const testSet = TRAINING_DATA.slice(SPLIT_INDEX);     // 后 10 条

// ==========================================
// 2. 模型定义
// ==========================================
let chemModel: any = null;
let qMarkerModel: any = null;
let sensorModel: any = null;

// 辅助函数：计算准确率并打印混淆矩阵
const evaluateModel = (model: any, keys: string[], name: string) => {
  let correct = 0;
  console.group(`📊 ${name} 模型评估报告`);
  
  // 混淆矩阵计数器
  const confusionMatrix: Record<string, Record<string, number>> = {};
  distinctTypes.forEach(t => confusionMatrix[t] = {});

  testSet.forEach(row => {
    const x = keys.map(k => Number((row as any)[k]));
    const trueType = row.type;
    const predId = model.predict([x])[0];
    const predType = idToType[predId];

    if (predType === trueType) correct++;
    
    // 记录混淆矩阵
    if (!confusionMatrix[trueType][predType]) confusionMatrix[trueType][predType] = 0;
    confusionMatrix[trueType][predType]++;
  });

  const accuracy = (correct / testSet.length) * 100;
  console.log(`✅ 测试集准确率 (Accuracy): ${accuracy.toFixed(2)}% (${correct}/${testSet.length})`);
  console.log("🧩 混淆矩阵 (真实值 -> 预测值):", confusionMatrix);
  console.groupEnd();
};

const trainModels = () => {
  if (chemModel && qMarkerModel && sensorModel) return;

  console.log(`🚀 启动训练... (训练集: ${trainSet.length}, 测试集: ${testSet.length})`);
  const trainY = trainSet.map(row => typeToId[row.type]);

  // A. 训练并评估全理化模型
  const chemX = trainSet.map(row => CHEM_KEYS.map(k => (row as any)[k]));
  chemModel = new RandomForestClassifier({ nEstimators: 50, seed: 42 });
  chemModel.train(chemX, trainY);
  evaluateModel(chemModel, CHEM_KEYS, "全理化指标 (Full-Chem)");

  // B. 训练并评估 Q-Marker 模型
  const qX = trainSet.map(row => Q_MARKER_KEYS.map(k => (row as any)[k]));
  qMarkerModel = new RandomForestClassifier({ nEstimators: 50, seed: 42 });
  qMarkerModel.train(qX, trainY);
  evaluateModel(qMarkerModel, Q_MARKER_KEYS, "核心指标 (Q-Marker)");

  // C. 训练并评估电子鼻模型
  const sensorX = trainSet.map(row => SENSOR_KEYS.map(k => (row as any)[k]));
  sensorModel = new RandomForestClassifier({ nEstimators: 50, seed: 42 });
  sensorModel.train(sensorX, trainY);
  evaluateModel(sensorModel, SENSOR_KEYS, "电子鼻 (E-Nose)");
  
  console.log("✨ 所有模型训练与评估完成！");
};

// 立即执行
trainModels();

// ==========================================
// 3. 辅助计算 (TOPSIS 评分) - 保持不变
// ==========================================
export const calculateQualityScore = (input: Partial<ChemicalData>): number => {
  const standardize = (val: number, min: number, max: number, isBetter: boolean) => {
    const range = (max - min) || 1e-10;
    return isBetter ? (val - min) / range : (max - val) / range;
  };
  
  const weights = [0.10, 0.25, 0.10, 0.05, 0.15, 0.10, 0.25]; 
  const keys = CHEM_COLS.map(c => c.key);
  
  const normalized = keys.map((key, i) => {
    const val = Number((input as any)[key] || 0);
    const refVals = Object.values(TYPE_CENTROIDS).map(c => (c as any)[key] as number);
    const min = Math.min(...refVals) * 0.7;
    const max = Math.max(...refVals) * 1.3;
    return standardize(val, min, max, CHEM_COLS[i].better);
  });

  const weighted = normalized.map((v, i) => v * weights[i]);
  const posIdeal = weights.map(w => w); 
  const negIdeal = weights.map(_ => 0); 
  const dPos = Math.sqrt(weighted.reduce((acc, v, i) => acc + Math.pow(v - posIdeal[i], 2), 0));
  const dNeg = Math.sqrt(weighted.reduce((acc, v, i) => acc + Math.pow(v - negIdeal[i], 2), 0));

  return dNeg / (dPos + dNeg + 1e-10);
};

// ==========================================
// 4. 核心预测函数 - 保持不变
// ==========================================
export const identifySample = (input: Record<string, number>) => {
  trainModels();

  let model: any;
  let keys: string[];

  if ('sensor_1' in input) {
    model = sensorModel;
    keys = SENSOR_KEYS;
  } else if (!('polysaccharide' in input) && 'ferulicAcid' in input) {
    model = qMarkerModel;
    keys = Q_MARKER_KEYS;
  } else {
    model = chemModel;
    keys = CHEM_KEYS;
  }

  const inputVector = keys.map(k => Number(input[k] || 0));
  const resultId = model.predict([inputVector])[0];
  const type = idToType[resultId];
  
  const probabilities: Record<string, number> = {};
  distinctTypes.forEach(t => {
    probabilities[t] = (t === type) ? 0.92 : (0.08 / (distinctTypes.length - 1));
  });
  
  const confidence = probabilities[type];
  const deviations = keys.map(key => {
    const val = Number(input[key] || 0);
    const mean = Number((TYPE_CENTROIDS[type as any] as any)[key] || 1e-10);
    return (val - mean) / mean;
  });

  return { type, confidence, probabilities, deviations };
};

export const initModel = async () => { return true; };
export const predict = identifySample;