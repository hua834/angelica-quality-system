
export type SampleType = '生当归' | '酒炙当归' | '酒洗当归' | '酒炒当归' | '酒浸当归';

export interface ChemicalData {
  polysaccharide: number;    // 多糖含量
  ferulicAcid: number;       // 阿魏酸
  totalAsh: number;          // 总灰分
  acidInsolubleAsh: number;  // 酸不溶性灰分
  volatileOil: number;       // 挥发油
  moisture: number;          // 水分
  extractContent: number;    // 浸出物
}

export interface PredictionResult {
  type: SampleType;
  confidence: number;
  qualityScore: number;
  deviations: number[];
  probabilities: Record<string, number>;
  qMarkers: string[];
}

export enum IdentificationMode {
  FULL_CHEM = 'FULL_CHEM',
  Q_MARKER = 'Q_MARKER',
  ENOSE = 'ENOSE'
}
