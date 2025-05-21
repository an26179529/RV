def calculate_diet_recommendations(nutrition_total):
    """
    根據每日建議攝取量計算飲食建議。
    
    參數:
        nutrition_total: 包含已檢測食物總營養值的字典
        
    回傳:
        包含剩餘建議攝取量和達成百分比的字典
    """
    # 標準每日建議攝取量（基於一般成人）
    daily_recommended = {
        'calories': 2000,      # 卡路里
        'protein': 60,         # 克
        'carbs': 300,          # 克
        'fiber': 25            # 克
    }
    
    recommendations = {}
    
    # 計算剩餘營養素和達成百分比
    for nutrient, recommended in daily_recommended.items():
        consumed = nutrition_total.get(nutrient, 0)
        remaining = max(0, recommended - consumed)
        percentage = min(100, round((consumed / recommended) * 100, 1)) if recommended > 0 else 0
        
        recommendations[nutrient] = {
            'consumed': consumed,
            'recommended': recommended,
            'remaining': remaining,
            'percentage': percentage
        }
    
    # 根據百分比生成具體建議
    diet_advice = []
    
    if recommendations['calories']['percentage'] < 25:
        diet_advice.append("這餐熱量較低，可適當增加其他食物的攝取。")
    elif recommendations['calories']['percentage'] > 40:
        diet_advice.append("這餐熱量攝取較高，建議今日其他餐點減少高熱量食物。")
    
    if recommendations['protein']['percentage'] < 20:
        diet_advice.append("蛋白質攝取不足，可考慮增加肉類、蛋、豆類等富含蛋白質的食物。")
    
    if recommendations['carbs']['percentage'] < 20:
        diet_advice.append("碳水化合物攝取不足，可適量增加全穀類、澱粉等食物。")
    elif recommendations['carbs']['percentage'] > 40:
        diet_advice.append("碳水化合物攝取較高，建議減少精緻澱粉類的攝取。")
    
    if recommendations['fiber']['percentage'] < 15:
        diet_advice.append("膳食纖維不足，建議增加蔬菜、水果與全穀類的攝取。")
    
    recommendations['advice'] = diet_advice
    return recommendations