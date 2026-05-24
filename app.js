// CardioShield Core Logic

let modelAssets = null;
let currentModel = 'lr'; // Default model is Logistic Regression

// Map internal feature keys to readable titles for UI
const featureLabels = {
  age: "Age",
  bmi: "Body Mass Index",
  systolic_bp: "Blood Pressure",
  cholesterol: "Cholesterol",
  glucose: "Blood Glucose",
  smoking: "Tobacco Use",
  exercise: "Physical Activity",
  stress: "Stress Level",
  diabetes: "Diabetes",
  family_history: "Family History"
};

// UI Elements
const els = {
  inputs: {
    age: document.getElementById('input-age'),
    bmi: document.getElementById('input-bmi'),
    sbp: document.getElementById('input-sbp'),
    chol: document.getElementById('input-chol'),
    gluc: document.getElementById('input-gluc'),
    stress: document.getElementById('input-stress'),
    exercise: document.getElementById('input-exercise'),
    smoking: document.getElementById('input-smoking'),
    diabetes: document.getElementById('input-diabetes'),
    family: document.getElementById('input-family')
  },
  vals: {
    age: document.getElementById('val-age'),
    bmi: document.getElementById('val-bmi'),
    sbp: document.getElementById('val-sbp'),
    chol: document.getElementById('val-chol'),
    gluc: document.getElementById('val-gluc'),
    stress: document.getElementById('val-stress'),
    exercise: document.getElementById('val-exercise')
  },
  btnModelLr: document.getElementById('btn-model-lr'),
  btnModelRf: document.getElementById('btn-model-rf'),
  gaugeFill: document.getElementById('gauge-risk-fill'),
  riskPercent: document.getElementById('display-risk-percent'),
  riskLabel: document.getElementById('display-risk-label'),
  riskDesc: document.getElementById('display-risk-desc'),
  heartIcon: document.getElementById('heart-pulse-icon'),
  contributionContainer: document.getElementById('contribution-container'),
  recommendationList: document.getElementById('recommendation-list'),
  tabCalc: document.getElementById('tab-calc'),
  tabInsights: document.getElementById('tab-insights'),
  btnTabCalc: document.getElementById('tab-calc-btn'),
  btnTabInsights: document.getElementById('tab-insights-btn')
};

// Start application
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const response = await fetch('./model_assets.json');
    if (!response.ok) throw new Error('Failed to load model asset config file.');
    modelAssets = await response.json();
    
    // Register slider events to update displayed numbers
    Object.keys(els.vals).forEach(key => {
      const inputEl = els.inputs[key === 'sbp' ? 'sbp' : key === 'chol' ? 'chol' : key === 'gluc' ? 'gluc' : key];
      if (inputEl) {
        inputEl.addEventListener('input', (e) => {
          let val = parseFloat(e.target.value);
          els.vals[key].textContent = key === 'bmi' ? val.toFixed(1) : val;
          calculateRisk();
        });
      }
    });

    // Checkboxes change events
    ['smoking', 'diabetes', 'family'].forEach(key => {
      els.inputs[key].addEventListener('change', calculateRisk);
    });

    // Model Selector Buttons
    els.btnModelLr.addEventListener('click', () => {
      currentModel = 'lr';
      els.btnModelLr.classList.add('active');
      els.btnModelRf.classList.remove('active');
      calculateRisk();
    });

    els.btnModelRf.addEventListener('click', () => {
      currentModel = 'rf';
      els.btnModelRf.classList.add('active');
      els.btnModelLr.classList.remove('active');
      calculateRisk();
    });

    // Tab Event Listeners
    els.btnTabCalc.addEventListener('click', () => {
      els.tabCalc.classList.add('active');
      els.tabInsights.classList.remove('active');
      els.btnTabCalc.classList.add('active');
      els.btnTabInsights.classList.remove('active');
    });

    els.btnTabInsights.addEventListener('click', () => {
      els.tabInsights.classList.add('active');
      els.tabCalc.classList.remove('active');
      els.btnTabInsights.classList.add('active');
      els.btnTabCalc.classList.remove('active');
    });

    // Run first prediction
    calculateRisk();

  } catch (err) {
    console.error('Initialization error:', err);
    els.riskDesc.textContent = "Error loading model weights. Please check your console.";
  }
});

// Traverses a single Decision Tree
function evaluateTree(node, features) {
  if (node.type === 'leaf') {
    return node.prob;
  }
  const featureVal = features[node.feature];
  if (featureVal <= node.threshold) {
    return evaluateTree(node.left, features);
  } else {
    return evaluateTree(node.right, features);
  }
}

// Compute Prediction Risk & UI Update
function calculateRisk() {
  if (!modelAssets) return;

  // Retrieve inputs
  const age = parseFloat(els.inputs.age.value);
  const bmi = parseFloat(els.inputs.bmi.value);
  const systolic_bp = parseFloat(els.inputs.sbp.value);
  const cholesterol = parseFloat(els.inputs.chol.value);
  const glucose = parseFloat(els.inputs.gluc.value);
  const stress = parseFloat(els.inputs.stress.value);
  const exercise = parseFloat(els.inputs.exercise.value);
  const smoking = els.inputs.smoking.checked ? 1 : 0;
  const diabetes = els.inputs.diabetes.checked ? 1 : 0;
  const family_history = els.inputs.family.checked ? 1 : 0;

  // Aggregate features object
  const features = {
    age,
    bmi,
    systolic_bp,
    cholesterol,
    glucose,
    smoking,
    exercise,
    stress,
    diabetes,
    family_history
  };

  let prob = 0;

  // Model Selection Evaluation
  if (currentModel === 'lr') {
    const lr = modelAssets.lr;
    let z = lr.intercept;
    for (const [feat, coef] of Object.entries(lr.coefficients)) {
      z += coef * features[feat];
    }
    // Sigmoid function
    prob = 1 / (1 + Math.exp(-z));
  } else {
    // Random Forest traversal
    const rf = modelAssets.rf;
    let totalProb = 0;
    rf.trees.forEach(tree => {
      totalProb += evaluateTree(tree, features);
    });
    prob = totalProb / rf.n_estimators;
  }

  updateUI(prob, features);
}

// Updates UI Elements
function updateUI(prob, features) {
  const percent = prob * 100;
  els.riskPercent.textContent = `${percent.toFixed(1)}%`;

  // Circular gauge display fill
  // Circle circumference is 2 * PI * r = 2 * 3.14159 * 50 = 314.15
  const circumference = 314.15;
  const offset = circumference * (1 - prob);
  els.gaugeFill.style.strokeDashoffset = offset;

  // Determine risk category & colors
  let category = "LOW RISK";
  let heartColorClass = "color-green";
  let pulseDuration = "1.8s";
  let gaugeColor = "#00f5d4"; // Teal
  let descText = "";

  if (prob < 0.35) {
    category = "LOW RISK";
    heartColorClass = "color-green";
    pulseDuration = "1.8s";
    gaugeColor = "hsl(165, 100%, 48%)";
    descText = "Your cardiovascular profile indicators are within safe ranges. Continue maintaining a balanced lifestyle, regular exercise, and standard health screenings.";
  } else if (prob < 0.70) {
    category = "MODERATE RISK";
    heartColorClass = "color-orange";
    pulseDuration = "1.0s";
    gaugeColor = "hsl(35, 100%, 50%)";
    descText = "Caution advised. Some physiological markers or lifestyle factors are elevating your heart risk profile. We recommend monitoring your blood pressure and checking cholesterol metrics.";
  } else {
    category = "HIGH RISK";
    heartColorClass = "color-red";
    pulseDuration = "0.5s";
    gaugeColor = "hsl(355, 100%, 55%)";
    descText = "High risk indicators detected. We strongly suggest scheduling a clinical consultation with a primary care physician or cardiologist to conduct comprehensive cardiac examinations.";
  }

  // Update classes
  els.riskLabel.textContent = category;
  els.riskLabel.className = `risk-label ${category.toLowerCase().replace(" ", "-")}`;
  els.riskLabel.style.color = gaugeColor;
  
  els.heartIcon.className = `heart-pulse-icon ${heartColorClass}`;
  els.heartIcon.style.setProperty('--pulse-duration', pulseDuration);
  els.gaugeFill.style.stroke = gaugeColor;
  els.riskDesc.textContent = descText;

  // Render feature contributions
  renderContributions(features);

  // Render lifestyle advice
  renderRecommendations(features);
}

// Compute individual feature impacts and render
function renderContributions(features) {
  if (!modelAssets) return;
  const lr = modelAssets.lr;
  const stats = modelAssets.feature_stats;
  
  let contributions = [];

  for (const [feat, coef] of Object.entries(lr.coefficients)) {
    const val = features[feat];
    const mean = stats[feat].mean;
    const diff = val - mean;
    const impact = coef * diff; // Feature contribution score relative to mean

    contributions.push({
      key: feat,
      label: featureLabels[feat],
      impact: impact,
      absImpact: Math.abs(impact)
    });
  }

  // Sort by absolute impact
  contributions.sort((a, b) => b.absImpact - a.absImpact);

  // Render rows
  els.contributionContainer.innerHTML = '';
  // Show top 5 features
  contributions.slice(0, 5).forEach(item => {
    const row = document.createElement('div');
    row.className = 'contrib-row';

    const label = document.createElement('span');
    label.className = 'contrib-label';
    label.textContent = item.label;

    const barWrapper = document.createElement('div');
    barWrapper.className = 'contrib-bar-wrapper';

    const bar = document.createElement('div');
    const isIncreasing = item.impact > 0;
    bar.className = `contrib-bar ${isIncreasing ? 'increasing' : 'decreasing'}`;
    
    // Scale bar size based on relative impact (max impact bounds around ~5 units in this logit scale)
    const maxScaleVal = 5.0;
    const percentWidth = Math.min((item.absImpact / maxScaleVal) * 100, 100);
    bar.style.width = `${percentWidth}%`;

    barWrapper.appendChild(bar);

    const impactText = document.createElement('span');
    impactText.className = `contrib-impact-text ${item.impact === 0 ? 'impact-neutral' : isIncreasing ? 'impact-high' : 'impact-low'}`;
    
    if (Math.abs(item.impact) < 0.05) {
      impactText.textContent = "Neutral";
      impactText.className = "contrib-impact-text impact-neutral";
    } else {
      impactText.textContent = isIncreasing ? "Elevates" : "Protects";
    }

    row.appendChild(label);
    row.appendChild(barWrapper);
    row.appendChild(impactText);

    els.contributionContainer.appendChild(row);
  });
}

// Generate lifestyle recommendations based on custom rules
function renderRecommendations(features) {
  els.recommendationList.innerHTML = '';
  const recommendations = [];

  if (features.smoking === 1) {
    recommendations.push({
      icon: "🚭",
      text: "<strong>Tobacco Cessation</strong>: Eliminating smoking decreases systemic vascular inflammation and reduces arterial disease risk instantly.",
      critical: true
    });
  }

  if (features.diabetes === 1) {
    recommendations.push({
      icon: "🩸",
      text: "<strong>Endocrine Check</strong>: Maintain close oversight of HbA1c levels. Keep dietary carbohydrate intakes balanced to manage insulin spikes.",
      critical: true
    });
  }

  if (features.systolic_bp > 130) {
    recommendations.push({
      icon: "🩺",
      text: `<strong>Hypertension Alert (${features.systolic_bp} mmHg)</strong>: Reduce daily sodium intake to <1,500mg and integrate heart-healthy cardiovascular conditioning.`,
      critical: features.systolic_bp > 150
    });
  }

  if (features.bmi > 25) {
    recommendations.push({
      icon: "⚖️",
      text: `<strong>BMI Management (${features.bmi.toFixed(1)})</strong>: Elevated BMI places higher load on cardiac ventricles. Aim for gradual caloric deficits.`,
      critical: features.bmi > 30
    });
  }

  if (features.exercise < 3) {
    recommendations.push({
      icon: "🏃‍♂️",
      text: "<strong>Active Lifestyle</strong>: Elevate physical conditioning. Strive for at least 150 minutes of light-to-moderate aerobic workouts weekly.",
      critical: false
    });
  }

  if (features.stress > 6) {
    recommendations.push({
      icon: "🧘",
      text: "<strong>Stress Decompression</strong>: High stress elevates vasoconstrictive cortisol. Allocate 10-15 minutes daily for breathing exercises or meditation.",
      critical: false
    });
  }

  if (features.cholesterol > 200) {
    recommendations.push({
      icon: "🥑",
      text: `<strong>Lipid Reduction (${features.cholesterol} mg/dL)</strong>: Incorporate soluble fiber (e.g. oats, beans) to bind cholesterol, and reduce saturated fats.`,
      critical: features.cholesterol > 240
    });
  }

  // Fallback if perfect health metrics are inputted
  if (recommendations.length === 0) {
    recommendations.push({
      icon: "🎉",
      text: "<strong>Perfect Vital Balance!</strong> All clinical features are in optimal ranges. Maintain your excellent physical habits.",
      critical: false
    });
  }

  // Render cards
  recommendations.forEach(rec => {
    const li = document.createElement('li');
    li.className = `rec-item ${rec.critical ? 'critical' : ''}`;
    
    const iconSpan = document.createElement('span');
    iconSpan.className = 'rec-icon';
    iconSpan.textContent = rec.icon;

    const textSpan = document.createElement('span');
    textSpan.innerHTML = rec.text;

    li.appendChild(iconSpan);
    li.appendChild(textSpan);
    els.recommendationList.appendChild(li);
  });
}
