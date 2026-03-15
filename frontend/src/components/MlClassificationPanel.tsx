import type { ClassificationModelResult, MethodologyStep, PriorityClassificationResult } from '../api/types';

type Props = {
  classification: PriorityClassificationResult;
};

const BAND_COLORS: Record<string, string> = {
  Rapido: '#e74c3c',
  Medio: '#f39c12',
  Largo: '#95a5a6',
  // Legacy 4-band colors
  Urgente: '#e74c3c',
  Corto: '#f59e0b',
};

const STEP_ICONS: Record<number, string> = {
  1: '1',
  2: '2',
  3: '3',
  4: '4',
  5: '5',
  6: '6',
  7: '7',
  8: '8',
  9: '9',
  10: '10',
};

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetric(value: number): string {
  return value.toFixed(4);
}

function BandCards({ classification }: Props) {
  return (
    <div className="mini-grid">
      {classification.bands.map((band) => {
        const color = BAND_COLORS[band.label] ?? '#95a5a6';
        return (
          <article
            key={band.label}
            className="mini-panel"
            style={{ borderLeft: `4px solid ${color}` }}
          >
            <strong style={{ color }}>{band.label}</strong>
            <p style={{ fontSize: '0.85rem' }}>
              {band.min_days}–{band.max_days > 999 ? '...' : band.max_days} dias
            </p>
            <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem' }}>
              <span>Train: <strong>{band.count_train}</strong></span>
              <span>Test: <strong>{band.count_test}</strong></span>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ClassifierTable({ models, bestModel }: { models: ClassificationModelResult[]; bestModel: string }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Modelo</th>
            <th>Accuracy</th>
            <th>Adj. Accuracy</th>
            <th>MAE (bandas)</th>
            <th>F1 Weighted</th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            const isWinner = model.model_name === bestModel && model.available;
            return (
              <tr
                key={model.model_name}
                style={isWinner ? { backgroundColor: 'rgba(15, 155, 114, 0.12)' } : undefined}
              >
                <td>
                  <div>{model.model_name}</div>
                  {isWinner ? (
                    <small style={{ color: '#0f9b72', fontWeight: 700 }}>Mejor modelo</small>
                  ) : null}
                </td>
                <td>{model.available ? formatPct(model.accuracy) : '-'}</td>
                <td>{model.available ? formatPct(model.adjacent_accuracy) : '-'}</td>
                <td>{model.available ? model.band_mae.toFixed(3) : '-'}</td>
                <td>{model.available ? formatMetric(model.f1_weighted) : '-'}</td>
                <td>
                  {model.available ? 'Disponible' : 'No disponible'}
                  {model.notes.length > 0 ? (
                    <small className="muted" style={{ display: 'block' }}>{model.notes[0]}</small>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ConfusionMatrix({ model, bandLabels }: { model: ClassificationModelResult; bandLabels: string[] }) {
  const matrix = model.confusion_matrix;
  if (!matrix || matrix.length === 0) return null;

  const maxVal = Math.max(1, ...matrix.flat());

  return (
    <div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Real \ Predicho</th>
              {bandLabels.map((lbl) => (
                <th key={lbl} style={{ color: BAND_COLORS[lbl] ?? undefined }}>{lbl}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={bandLabels[i]}>
                <td style={{ fontWeight: 600, color: BAND_COLORS[bandLabels[i]] ?? undefined }}>
                  {bandLabels[i]}
                </td>
                {row.map((cell, j) => {
                  const isDiagonal = i === j;
                  const alpha = cell / maxVal;
                  return (
                    <td
                      key={j}
                      className="pivot-value-cell"
                      style={{
                        '--heat-alpha': alpha,
                        textAlign: 'center',
                        fontWeight: isDiagonal ? 700 : 400,
                        backgroundColor: isDiagonal
                          ? `rgba(15, 155, 114, ${0.15 + alpha * 0.45})`
                          : `rgba(207, 92, 54, ${alpha * 0.3})`,
                        color: alpha > 0.7 ? '#fff' : 'var(--text)',
                      } as React.CSSProperties}
                    >
                      {cell}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {model.per_class.length > 0 ? (
        <div className="table-wrap" style={{ marginTop: '1rem' }}>
          <table>
            <thead>
              <tr>
                <th>Banda</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {model.per_class.map((cls) => (
                <tr key={cls.band}>
                  <td style={{ color: BAND_COLORS[cls.band] ?? undefined, fontWeight: 600 }}>{cls.band}</td>
                  <td>{formatMetric(cls.precision)}</td>
                  <td>{formatMetric(cls.recall)}</td>
                  <td>{formatMetric(cls.f1)}</td>
                  <td>{cls.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

function MethodologyStepCard({ step }: { step: MethodologyStep }) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: '10px',
        padding: '1.25rem 1.5rem',
        marginBottom: '1rem',
        backgroundColor: 'var(--bg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
        <div
          style={{
            minWidth: '36px',
            height: '36px',
            borderRadius: '50%',
            backgroundColor: '#21495f',
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 700,
            fontSize: '0.85rem',
            flexShrink: 0,
          }}
        >
          {STEP_ICONS[step.step] ?? step.step}
        </div>
        <div style={{ flex: 1 }}>
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem', color: 'var(--text)' }}>
            {step.title}
          </h4>

          <div style={{ marginBottom: '0.75rem' }}>
            <div
              style={{
                fontSize: '0.78rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: '#64748b',
                marginBottom: '0.25rem',
              }}
            >
              Por que
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.6, color: 'var(--text)' }}>
              {step.rationale}
            </p>
          </div>

          <div style={{ marginBottom: step.evidence ? '0.75rem' : 0 }}>
            <div
              style={{
                fontSize: '0.78rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: '#0f9b72',
                marginBottom: '0.25rem',
              }}
            >
              Decision
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.6, color: 'var(--text)', fontWeight: 500 }}>
              {step.decision}
            </p>
          </div>

          {step.evidence ? (
            <div>
              <div
                style={{
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: '#cf5c36',
                  marginBottom: '0.25rem',
                }}
              >
                Evidencia
              </div>
              <p
                style={{
                  margin: 0,
                  fontSize: '0.85rem',
                  lineHeight: 1.6,
                  color: 'var(--muted)',
                  backgroundColor: 'rgba(0,0,0,0.03)',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  borderLeft: '3px solid #cf5c36',
                }}
              >
                {step.evidence}
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

type FeatureGroupDef = {
  title: string;
  color: string;
  bgColor: string;
  description: string;
  example: string;
  features: string[];
};

function classifyFeatures(featureNames: string[]): FeatureGroupDef[] {
  const targetEncSimple = featureNames.filter(
    (f) => f.endsWith('_target_enc') && !f.includes('x') && !f.includes('lag'),
  );
  const targetEncInteraction = featureNames.filter(
    (f) => f.endsWith('_target_enc') && f.includes('x') && !f.includes('lag'),
  );
  const freqEnc = featureNames.filter((f) => f.endsWith('_freq'));
  const countEnc = featureNames.filter((f) => f.endsWith('_count'));
  const lagMedian = featureNames.filter((f) => f.endsWith('_lag_median'));
  const lagStd = featureNames.filter((f) => f.endsWith('_lag_std'));
  const lagTrend = featureNames.filter((f) => f.endsWith('_lag_trend'));
  const lagVolume = featureNames.filter((f) => f.endsWith('_lag_volume'));
  const global = featureNames.filter((f) => f === 'prev_week_volume' || f === 'week_number');

  const groups: FeatureGroupDef[] = [];

  if (targetEncSimple.length > 0)
    groups.push({
      title: 'Target encoding simple',
      color: '#21495f',
      bgColor: 'rgba(33, 73, 95, 0.08)',
      description:
        'Mediana de DaysInDeposit por cada categoria individual, calculada sobre el conjunto de entrenamiento. Convierte una categoria (ej: Owner=7) en un numero que representa "cuanto se quedan tipicamente los contenedores de ese grupo".',
      example:
        'Owner_target_enc = 3.0 significa que la mediana de todos los contenedores de Owner 7 en train fue 3 dias. Size_target_enc = 6.0 para Size=2 vs 26.5 para Size=1.',
      features: targetEncSimple,
    });

  if (targetEncInteraction.length > 0)
    groups.push({
      title: 'Target encoding de interacciones',
      color: '#21495f',
      bgColor: 'rgba(33, 73, 95, 0.08)',
      description:
        'Mediana de DaysInDeposit por cada PAR de categorias. Captura que la combinacion importa: un contenedor DRY de Owner 7 se comporta distinto a un RF de Owner 1, aunque individualmente Owner y Type no lo expliquen.',
      example:
        'OwnerxQuality_target_enc: Owner=7 + CLASE B-C tiene mediana 2.5 dias, pero Owner=1 + INSPECTION tiene mediana 55 dias. Esta interaccion tiene correlacion 0.496 con el target, mas alta que cualquier feature individual.',
      features: targetEncInteraction,
    });

  if (freqEnc.length > 0)
    groups.push({
      title: 'Frequency encoding',
      color: '#6366f1',
      bgColor: 'rgba(99, 102, 241, 0.08)',
      description:
        'Proporcion de cada categoria en train (valor entre 0.0 y 1.0). Los owners mas frecuentes tienden a retirar contenedores mas rapido porque tienen operaciones regulares y booking predecible.',
      example:
        'Owner_freq = 0.49 para Owner 7 (el mas comun, mueve rapido) vs Owner_freq = 0.01 para Owner 4 (poco frecuente, tiempos largos). Correlacion con target: -0.400.',
      features: freqEnc,
    });

  if (countEnc.length > 0)
    groups.push({
      title: 'Count encoding',
      color: '#6366f1',
      bgColor: 'rgba(99, 102, 241, 0.08)',
      description:
        'Cantidad absoluta de registros de esa categoria en train. Similar al frequency encoding pero en escala absoluta, util cuando el volumen total importa.',
      example:
        'Owner_count = 3600 para Owner 7 vs Owner_count = 80 para Owner 4. Owners con alto volumen operan regularmente y retiran mas rapido.',
      features: countEnc,
    });

  if (lagMedian.length > 0)
    groups.push({
      title: 'Lag median (historia acumulada)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Mediana de DaysInDeposit del grupo en TODAS las semanas anteriores al registro actual. Para un contenedor en semana 5, usa datos de semanas 1-4. No trackea el mismo contenedor: usa el comportamiento agregado del grupo.',
      example:
        'Owner_lag_median para Owner=7 en semana 5: mediana historica de 3 dias (basada en ~3200 contenedores de semanas 1-4). La correlacion con dias reales mejora con mas historia: 1 semana→0.03, 4 semanas→0.59.',
      features: lagMedian,
    });

  if (lagStd.length > 0)
    groups.push({
      title: 'Lag std (volatilidad historica)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Desviacion estandar historica de DaysInDeposit del grupo. Mide que tan predecible es un grupo: un Owner con std baja es confiable, uno con std alta es impredecible.',
      example:
        'Owner_lag_std = 2.5 para Owner 7 (estable, siempre retira en ~3 dias) vs Owner_lag_std = 20.1 para Owner 4 (impredecible, puede ser 2 dias o 46 dias). Grupos con alta volatilidad son mas dificiles de clasificar.',
      features: lagStd,
    });

  if (lagTrend.length > 0)
    groups.push({
      title: 'Lag trend (tendencia reciente)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Diferencia de mediana entre las 2 semanas mas recientes. Captura si un grupo esta acelerando o frenando su retiro. Valor positivo = se estan quedando mas, negativo = se estan yendo mas rapido.',
      example:
        'Owner_lag_trend = -3.0 para Owner 7: la mediana bajo 3 dias entre semana 3 y 4, esta retirando mas rapido. Owner_lag_trend = +15 para Owner 4: salto de 15 dias, algo cambio (buque atrasado?).',
      features: lagTrend,
    });

  if (lagVolume.length > 0)
    groups.push({
      title: 'Lag volume (actividad historica)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Cantidad acumulada de contenedores del grupo en semanas anteriores. Proxy de nivel de actividad y confiabilidad de las estadisticas: mas volumen = estimaciones mas estables.',
      example:
        'Owner_lag_volume = 3200 para Owner 7 (muchos datos, estimacion confiable) vs Owner_lag_volume = 15 para Owner 9 (pocos datos, estimacion ruidosa).',
      features: lagVolume,
    });

  if (global.length > 0)
    groups.push({
      title: 'Variables globales',
      color: '#cf5c36',
      bgColor: 'rgba(207, 92, 54, 0.08)',
      description:
        'Features que no dependen de una categoria especifica sino del contexto general del puerto.',
      example:
        'prev_week_volume: total de contenedores la semana anterior (proxy de congestion portuaria — si entran muchos, los tiempos suben). week_number: numero de semana, captura tendencia lineal o estacionalidad.',
      features: global,
    });

  return groups;
}

function FeatureGroupCard({ group }: { group: FeatureGroupDef }) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: '8px',
        padding: '1rem',
        marginBottom: '0.75rem',
        borderLeft: `4px solid ${group.color}`,
        backgroundColor: group.bgColor,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.5rem' }}>
        <h5 style={{ margin: 0, fontSize: '0.9rem', color: group.color }}>{group.title}</h5>
        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{group.features.length} variables</span>
      </div>

      <p style={{ margin: '0 0 0.5rem', fontSize: '0.85rem', lineHeight: 1.6, color: 'var(--text)' }}>
        {group.description}
      </p>

      <div
        style={{
          fontSize: '0.82rem',
          lineHeight: 1.6,
          color: 'var(--muted)',
          backgroundColor: 'rgba(255,255,255,0.5)',
          padding: '0.5rem 0.75rem',
          borderRadius: '5px',
          marginBottom: '0.5rem',
          fontStyle: 'italic',
        }}
      >
        Ej: {group.example}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
        {group.features.map((f) => (
          <code
            key={f}
            style={{
              fontSize: '0.72rem',
              padding: '0.15rem 0.4rem',
              borderRadius: '3px',
              backgroundColor: 'rgba(255,255,255,0.6)',
              border: `1px solid ${group.color}33`,
              color: group.color,
            }}
          >
            {f}
          </code>
        ))}
      </div>
    </div>
  );
}

function MethodologySection({ classification }: Props) {
  const methodology = classification.methodology;
  if (!methodology || methodology.length === 0) return null;

  const featureNames = classification.feature_names ?? [];
  const featureGroups = classifyFeatures(featureNames);

  return (
    <div className="panel table-panel">
      <div className="table-header">
        <div>
          <h3>Metodologia y justificacion</h3>
          <span className="muted">
            Detalle paso a paso de cada decision tomada en el pipeline de clasificacion.
          </span>
        </div>
      </div>
      <div style={{ padding: '1rem 1.5rem 1.5rem' }}>

        {/* Origin explanation */}
        <div
          style={{
            marginBottom: '1.5rem',
            padding: '1.25rem',
            borderRadius: '8px',
            border: '1px solid var(--border)',
            backgroundColor: 'rgba(33, 73, 95, 0.04)',
            borderLeft: '4px solid #21495f',
          }}
        >
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem' }}>
            De donde salen las {featureNames.length} variables?
          </h4>
          <p style={{ margin: '0 0 0.75rem', fontSize: '0.9rem', lineHeight: 1.7 }}>
            El dataset original solo tiene <strong>4 columnas categoricas</strong> (Owner, Size, Type, Quality),
            el target (DaysInDeposit) y la semana. Un arbol de decision no puede usar categorias directamente.
            Las {featureNames.length} variables se <strong>construyen matematicamente</strong> a partir de esas
            4 columnas — no son datos nuevos, son <strong>transformaciones</strong> que extraen la senal
            que ya existe en los datos pero en una forma que los modelos pueden procesar.
          </p>
          <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.7 }}>
            Ejemplo: para un contenedor de <code>Owner=7, Size=2, Type=DRY</code> en semana 5, se calcula:
            "la mediana historica de Owner 7 fue 3 dias" (<code>Owner_target_enc=3.0</code>),
            "Owner 7 representa el 49% del trafico" (<code>Owner_freq=0.49</code>),
            "en semanas 1-4 la mediana de Owner 7 fue 3 dias" (<code>Owner_lag_median=3.0</code>),
            "la tendencia bajo 3 dias" (<code>Owner_lag_trend=-3.0</code>). Todo calculado sin mirar la
            semana que se predice (sin leakage temporal).
          </p>
        </div>

        {/* Feature inventory by group */}
        {featureGroups.length > 0 ? (
          <div style={{ marginBottom: '1.5rem' }}>
            <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.95rem' }}>
              Inventario de features ({featureNames.length} variables en {featureGroups.length} grupos)
            </h4>
            {featureGroups.map((group) => (
              <FeatureGroupCard key={group.title} group={group} />
            ))}
          </div>
        ) : null}

        {/* Priority score summary */}
        {classification.priority_score_corr != null ? (
          <div
            style={{
              marginBottom: '1.5rem',
              padding: '1rem',
              borderRadius: '8px',
              border: '1px solid var(--border)',
              backgroundColor: 'rgba(15, 155, 114, 0.04)',
              borderLeft: '4px solid #0f9b72',
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#0f9b72' }}>
              Priority Score para el optimizador
            </h4>
            <p style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', lineHeight: 1.6 }}>
              score = P(Rapido) x 1 + P(Medio) x 2 + P(Largo) x 3
            </p>
            <div className="mini-grid">
              <article className="mini-panel">
                <strong>Correlacion vs dias reales</strong>
                <p>{classification.priority_score_corr.toFixed(3)}</p>
              </article>
              {classification.priority_score_stats?.min != null ? (
                <article className="mini-panel">
                  <strong>Rango</strong>
                  <p>[{classification.priority_score_stats.min}, {classification.priority_score_stats.max}]</p>
                </article>
              ) : null}
              {classification.priority_score_stats?.mean != null ? (
                <article className="mini-panel">
                  <strong>Media / Std</strong>
                  <p>{classification.priority_score_stats.mean} / {classification.priority_score_stats.std}</p>
                </article>
              ) : null}
            </div>
          </div>
        ) : null}

        {/* Steps */}
        {methodology.map((step) => (
          <MethodologyStepCard key={step.step} step={step} />
        ))}
      </div>
    </div>
  );
}

export default function MlClassificationPanel({ classification }: Props) {
  const bestModel = classification.models.find((m) => m.model_name === classification.best_model && m.available);
  const bandLabels = classification.bands.map((b) => b.label);

  return (
    <section className="stack">
      {/* Hero / Narrative */}
      <div className="panel" style={{ borderLeft: '4px solid #0f9b72' }}>
        <div className="section-header">
          <div>
            <h3>Clasificacion por bandas de prioridad</h3>
            <p style={{ fontSize: '1.05rem', lineHeight: 1.6, margin: '0.5rem 0 0' }}>
              {classification.narrative}
            </p>
            <p className="muted" style={{ marginTop: '0.5rem' }}>
              Baseline (clase mas frecuente): {formatPct(classification.baseline_accuracy)}
              {classification.priority_score_corr != null ? (
                <> | Corr. priority score vs dias: {classification.priority_score_corr.toFixed(3)}</>
              ) : null}
            </p>
          </div>
        </div>
      </div>

      {/* Band distribution cards */}
      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Bandas de prioridad</h3>
            <span className="muted">Distribucion de contenedores por banda en train y test.</span>
          </div>
        </div>
        <div style={{ padding: '1rem 1.5rem 1.5rem' }}>
          <BandCards classification={classification} />
        </div>
      </div>

      {/* Classifier comparison table */}
      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Modelos clasificadores</h3>
            <span className="muted">
              Accuracy = acierto exacto de banda. Adj. Accuracy = acierto o error por 1 banda adyacente.
            </span>
          </div>
        </div>
        <ClassifierTable models={classification.models} bestModel={classification.best_model} />
      </div>

      {/* Confusion matrix of best model */}
      {bestModel ? (
        <div className="panel table-panel">
          <div className="table-header">
            <div>
              <h3>Matriz de confusion — {bestModel.model_name}</h3>
              <span className="muted">
                Filas = banda real, columnas = banda predicha. Diagonal = aciertos.
              </span>
            </div>
          </div>
          <div style={{ padding: '1rem 1.5rem 1.5rem' }}>
            <ConfusionMatrix model={bestModel} bandLabels={bandLabels} />
          </div>
        </div>
      ) : null}

      {/* Methodology section */}
      <MethodologySection classification={classification} />
    </section>
  );
}
