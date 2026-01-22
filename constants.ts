import { SampleType } from './types';

export const CHEM_COLS = [
  { key: 'polysaccharide', label: '多糖含量 (mg/g)', better: true, standardRange: '> 30.0' },
  { key: 'ferulicAcid', label: '阿魏酸含量 (%)', better: true, standardRange: '≥ 0.05' },
  { key: 'totalAsh', label: '总灰分 (%)', better: false, standardRange: '≤ 7.0' },
  { key: 'acidInsolubleAsh', label: '酸不溶性灰分 (%)', better: false, standardRange: '≤ 2.0' },
  { key: 'volatileOil', label: '挥发油含量 (mL/g)', better: true, standardRange: '≥ 0.4' },
  { key: 'moisture', label: '水分 (%)', better: false, standardRange: '≤ 15.0' },
  { key: 'extractContent', label: '浸出物含量 (%)', better: true, standardRange: '≥ 45.0' },
];

export const SENSOR_COLS = [
  { key: 'sensor_1', code: 'W1C', label: '芳烃化合物', substance: '芳烃化合物' },
  { key: 'sensor_2', code: 'W5S', label: '氮氧化合物', substance: '氮氧化合物' },
  { key: 'sensor_3', code: 'W3C', label: '氨/芳香分子', substance: '氨，芳香分子' },
  { key: 'sensor_4', code: 'W6S', label: '氢化物', substance: '氢化物' },
  { key: 'sensor_5', code: 'W5C', label: '烯烃/芳族', substance: '烯烃、芳族，极性分子' },
  { key: 'sensor_6', code: 'W1S', label: '烷类', substance: '烷类' },
  { key: 'sensor_7', code: 'W1W', label: '硫化合物', substance: '硫化合物' },
  { key: 'sensor_8', code: 'W2S', label: '醇类/芳香族', substance: '醇类，部分芳香族化合物' },
  { key: 'sensor_9', code: 'W2W', label: '硫有机物', substance: '芳烃化合物，硫的有机化合物' },
  { key: 'sensor_10', code: 'W3S', label: '烷类/脂肪族', substance: '烷类和脂肪族' },
];

export const SAMPLE_TYPES: SampleType[] = ['生当归', '酒炙当归', '酒洗当归', '酒炒当归', '酒浸当归'];

/**
 * 根据 PEN3 实验数据图片提取的中心值 (Centroids)
 */
export const TYPE_CENTROIDS: Record<SampleType, any> = {
  '生当归': { 
    polysaccharide: 43.98, ferulicAcid: 0.0907, totalAsh: 5.65, acidInsolubleAsh: 0.458, volatileOil: 0.496, moisture: 8.63, extractContent: 48.36,
    sensor_1: 0.3957, sensor_2: 2.112, sensor_3: 0.4411, sensor_4: 1.226, sensor_5: 0.485, sensor_6: 2.332, sensor_7: 11.09, sensor_8: 2.062, sensor_9: 1.021, sensor_10: 1.701
  },
  '酒炙当归': { 
    polysaccharide: 41.98, ferulicAcid: 0.0862, totalAsh: 4.51, acidInsolubleAsh: 0.626, volatileOil: 0.542, moisture: 8.55, extractContent: 48.34,
    sensor_1: 0.4779, sensor_2: 2.068, sensor_3: 0.5001, sensor_4: 1.189, sensor_5: 0.527, sensor_6: 2.172, sensor_7: 9.921, sensor_8: 1.911, sensor_9: 1.022, sensor_10: 1.552
  },
  '酒洗当归': { 
    polysaccharide: 59.78, ferulicAcid: 0.0814, totalAsh: 5.01, acidInsolubleAsh: 0.456, volatileOil: 0.448, moisture: 11.23, extractContent: 47.76,
    sensor_1: 0.4941, sensor_2: 1.913, sensor_3: 0.526, sensor_4: 1.191, sensor_5: 0.555, sensor_6: 2.121, sensor_7: 8.291, sensor_8: 1.838, sensor_9: 1.018, sensor_10: 1.484
  },
  '酒炒当归': { 
    polysaccharide: 51.24, ferulicAcid: 0.0901, totalAsh: 4.96, acidInsolubleAsh: 0.442, volatileOil: 0.484, moisture: 7.72, extractContent: 53.64,
    sensor_1: 0.4498, sensor_2: 2.401, sensor_3: 0.475, sensor_4: 1.234, sensor_5: 0.498, sensor_6: 2.364, sensor_7: 12.44, sensor_8: 2.003, sensor_9: 1.042, sensor_10: 1.553
  },
  '酒浸当归': { 
    polysaccharide: 46.46, ferulicAcid: 0.0874, totalAsh: 5.82, acidInsolubleAsh: 0.522, volatileOil: 0.485, moisture: 12.82, extractContent: 28.34,
    sensor_1: 0.5019, sensor_2: 2.051, sensor_3: 0.518, sensor_4: 1.161, sensor_5: 0.538, sensor_6: 2.109, sensor_7: 9.482, sensor_8: 1.789, sensor_9: 1.044, sensor_10: 1.482
  }
};

export const REFERENCE_STD = {
  polysaccharide: 8.0,
  ferulicAcid: 0.005,
  totalAsh: 0.6,
  acidInsolubleAsh: 0.08,
  volatileOil: 0.06,
  moisture: 1.5,
  extractContent: 6.0,
  sensor: 0.08
};