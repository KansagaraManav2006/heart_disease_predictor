export const generateSuggestions = (type, data, predictionResult) => {
    const suggestions = [];
    const isHighRisk = predictionResult.prediction === 1;

    if (type === 'diabetes') {
        const hba1c = Number(data.hba1c);
        const glucose = Number(data.glucose);
        const bmi = Number(data.bmi);
        const smoking = String(data.smokingHistory).toLowerCase();

        if (glucose > 125) {
            suggestions.push("Your fasting blood glucose is elevated. Target levels are generally below 100 mg/dL. Reduce refined sugar intake and monitor your blood glucose regularly.");
        } else if (glucose > 100) {
            suggestions.push("Your glucose is in the prediabetic range. Focus on a whole-food, low-glycemic index diet.");
        }

        if (hba1c >= 6.5) {
            suggestions.push("HbA1c levels indicate higher long-term blood sugar levels. Consult an endocrinologist for a customized management plan.");
        } else if (hba1c >= 5.7) {
            suggestions.push("HbA1c levels are slightly elevated. Regular exercise (150 mins/week) can drastically improve insulin sensitivity.");
        }

        if (bmi >= 30) {
            suggestions.push("Your BMI indicates obesity, a primary risk factor. Try to implement a sustainable weight management plan incorporating balanced meals and daily activity.");
        } else if (bmi >= 25) {
            suggestions.push("Your BMI is in the overweight category. Gradual weight reduction of 5-10% can significantly improve metabolic outcomes.");
        }

        if (['current', 'ever'].includes(smoking)) {
            suggestions.push("Smoking increases insulin resistance. Ceasing smoking will have an immediate positive impact on your cardiovascular and metabolic health.");
        }

    } else if (type === 'heart') {
        const sys = Number(data.systolic_bp);
        const chol = Number(data.cholesterol);
        const age = Number(data.age);
        // Support both string ('1'/'0') and boolean values for lifestyle flags
        const smoke = data.smoke === '1' || data.smoke === true || data.smoke === 1;
        const alco = data.alco === '1' || data.alco === true || data.alco === 1;
        const active = data.active === '1' || data.active === true || data.active === 1;

        if (sys >= 140) {
            suggestions.push("Blood pressure is elevated (≥140 mmHg). Reduce sodium intake to under 1,500mg daily, manage stress, and consider the DASH diet.");
        } else if (sys >= 130) {
            suggestions.push("Blood pressure is slightly elevated. Lifestyle modifications including regular aerobic exercise are recommended.");
        }

        if (chol > 240) {
            suggestions.push("High cholesterol detected. Replace saturated fats with healthy fats (like avocados, nuts) and increase dietary fiber from oats and legumes.");
        } else if (chol >= 200) {
            suggestions.push("Borderline high cholesterol. Moderate your intake of animal fats and processed foods.");
        }

        if (smoke) {
            suggestions.push("Smoking significantly increases cardiovascular risk. Quitting smoking is one of the most impactful steps you can take for your heart health.");
        }

        if (alco) {
            suggestions.push("Regular alcohol consumption increases blood pressure and heart risk. Consider reducing or eliminating alcohol intake.");
        }

        if (!active) {
            suggestions.push("Physical inactivity is a major cardiac risk factor. Aim for at least 150 minutes of moderate aerobic exercise per week.");
        }

        if (age >= 55) {
            suggestions.push("Regular cardiac screening is especially important at your age. Annual check-ups with a cardiologist are highly recommended.");
        }
    }

    // General suggestions
    if (isHighRisk && suggestions.length < 3) {
        suggestions.push("Prioritize a comprehensive clinical evaluation. The system detected high-risk complex patterns.");
    }

    if (suggestions.length === 0 && !isHighRisk) {
        suggestions.push("Your biometric indicators look exceptional. Continue maintaining your current active lifestyle and balanced diet!");
    }

    return suggestions;
};
