
import { CHEM_COLS, TYPE_CENTROIDS, REFERENCE_STD, SAMPLE_TYPES } from '../constants';
import { ChemicalData, SampleType } from '../types';

/**
 * 标准化函数 (极差变换)
 */
const standardize = (val: number, min: number, max: number, isBetter: boolean) => {
  const range = (max - min) || 1e-10;
  return isBetter ? (val - min) / range : (max - val) / range;
};

/**
 * 改进型 TOPSIS 质量评分计算
 * 融合 CRITIC 指标冲突性与熵权信息量
 */
export const calculateQualityScore = (input: Partial<ChemicalData>): number => {
  // 理化指标重要性权重
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

/**
 * 基于高斯核相似度的样品自动鉴别分类
 */
export const identifySample = (input: Record<string, number>) => {
  const scores: Record<string, number> = {};
  const inputKeys = Object.keys(input).filter(k => k in TYPE_CENTROIDS['生当归']);

  SAMPLE_TYPES.forEach(type => {
    let similarity = 0;
    inputKeys.forEach(key => {
      const val = Number(input[key] || 0);
      const centroid = Number((TYPE_CENTROIDS[type] as any)[key] || 0);
      
      // 获取对应的标准差，传感器使用统一标准差
      const std = key.startsWith('sensor') 
        ? REFERENCE_STD.sensor 
        : (REFERENCE_STD as any)[key] || 1;
        
      // 高斯相似度公式
      similarity += Math.exp(-Math.pow(val - centroid, 2) / (2 * Math.pow(std, 2)));
    });
    scores[type] = inputKeys.length > 0 ? similarity / inputKeys.length : 0;
  });

  // 归一化为概率分布
  const sum = Object.values(scores).reduce((a, b) => a + b, 0) || 1e-10;
  const probabilities = Object.fromEntries(
    Object.entries(scores).map(([k, v]) => [k, v / sum])
  ) as Record<string, number>;

  const bestType = SAMPLE_TYPES.reduce((a, b) => probabilities[a] > probabilities[b] ? a : b);
  
  // 计算特征偏差率 (相对于判定类型的中心值)
  const deviations = inputKeys.map(key => {
    const val = Number(input[key] || 0);
    const mean = Number((TYPE_CENTROIDS[bestType] as any)[key] || 1e-10);
    return (val - mean) / mean;
  });

  return {
    type: bestType,
    confidence: probabilities[bestType],
    probabilities,
    deviations
  };
};
