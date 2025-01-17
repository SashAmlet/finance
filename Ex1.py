import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import RadioButtons
sys.stdout.reconfigure(encoding='utf-8')

def export_to_excel(historical_returns, tickers, filename='historical_returns.xlsx'):
    # Перетворення масиву в DataFrame
    df = pd.DataFrame(historical_returns, columns=tickers)
    
    # Запис DataFrame в Excel
    df.to_excel(filename, index=False)
    
    print(f"Дані успішно експортовані в файл {filename}")

def plot_histograms(returns, num_bins=50):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.3)
    
    # Створення радіо-кнопок для вибору акції
    ax_radio = plt.axes([0.05, 0.4, 0.2, 0.2], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(ax_radio, returns.columns)
    
    def update_histogram(label):
        ax.clear()
        ticker_returns = returns[label]
        max_return = ticker_returns.max()
        min_return = ticker_returns.min()
        bins = np.linspace(min_return, max_return, num_bins)
        
        ax.hist(ticker_returns, bins=bins, edgecolor='black')
        ax.set_title(f'Гістограма доходностей для {label}')
        ax.set_xlabel('Доходність')
        ax.set_ylabel('Частота')
        ax.grid(True)
        fig.canvas.draw_idle()
    
    radio.on_clicked(update_histogram)
    
    # Відображення гістограми для першої акції за замовчуванням
    update_histogram(returns.columns[0])
    
    plt.show()

def get_historical_returns(tickers, start_date, end_date):
    # Завантаження даних про ціни акцій
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Перевірка, чи дані завантажилися коректно
    if data.empty:
        print("Дані не завантажилися. Перевірте тикери та дати.")
        return np.array([])
    
    # Обчислення щоденних доходностей (у відсотках) на основі зміни ціни на акції
    clean_data = data.dropna()
    returns = (clean_data / clean_data.shift(1) - 1).iloc[1:]
    
    return returns

def optimize_portfolio(mean_returns, cov_matrix, target_return):
    # Цільова функція (мінімізація ризику)   
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(weights.T @ (cov_matrix @ weights))

    # Умови
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Сума ваг = 1
        {'type': 'eq', 'fun': lambda w: (w @ mean_returns) - target_return}  # Доходність = target_return
    ]

    # Обмеження на ваги
    bounds = [(0, 1) for _ in range(len(mean_returns))]

    # Початкове наближення
    initial_weights = np.ones(len(mean_returns)) / len(mean_returns)

    # Оптимізація
    result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

def logarithmization(data, target_return, mean_returns):
    data = np.log(data + 1)
    target_return = np.log(target_return + 1)
    mean_returns = np.log(mean_returns + 1)

    return data, target_return, mean_returns

def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_portfolios)
    efficient_portfolios = []

    for target_return in target_returns:
        result = optimize_portfolio(mean_returns, cov_matrix, target_return)
        if result.success:
            efficient_portfolios.append((result.fun, target_return))
    
    risks, returns = zip(*efficient_portfolios)
    
    plt.figure(figsize=(10, 6))
    plt.plot(risks, returns, 'o-', markersize=5, label='Efficient Frontier')
    plt.title('Efficient Frontier')
    plt.xlabel('Risk (Volatility)')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.show()


# Data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SQQQ'] #, 'FB', 'JPM', 'JNJ', 'V', 'PG', 'DIS']
start_date = '2022-01-01'
end_date = '2023-01-01'

historical_returns = get_historical_returns(tickers, start_date, end_date)
mean_returns = np.mean(historical_returns, axis=0)  # Ожидаемые доходности активов
target_return = 0.0015    # Целевая доходность портфеля

# export_to_excel(historical_returns, tickers)

# Logarithmizing the data
log_historical_returns, log_target_return, log_mean_returns = logarithmization(historical_returns, target_return, mean_returns)

cov_matrix = np.cov(log_historical_returns, rowvar=False)


result = optimize_portfolio(log_mean_returns, cov_matrix, log_target_return)

# Output results
print("Ожидаемые доходности активов:", mean_returns)
print("Оптимальные веса активов:", result.x)
print("Минимальный риск (волатильность):", result.fun)  

plot_histograms(historical_returns)

plot_efficient_frontier(log_mean_returns, cov_matrix)