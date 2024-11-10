from flask import Flask, render_template, request
import numpy as np
from scipy.stats import norm
import yfinance as yf

app = Flask(__name__)

def blackScholes(r, S, K, T, sigma, type="C"):
    """Calculate Black-Scholes option price for a call/put."""
    try:
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if type == "C":  # Call option
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif type == "P":  # Put option
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Enter 'C' for Call or 'P' for Put.")
        return price
    except Exception as e:
        print("Error in calculating option price:", e)
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        ticker_symbol = request.form['ticker'].upper()
        try:
            K = float(request.form['strike'])
            if K <= 0:
                raise ValueError("Strike price must be positive.")
        except ValueError:
            error = "Invalid strike price. Please enter a positive number."
            return render_template('index.html', error=error)

        try:
            T_days = int(request.form['time'])
            if T_days <= 0:
                raise ValueError("Time to expiration must be positive.")
        except ValueError:
            error = "Invalid time to expiration. Please enter a positive integer."
            return render_template('index.html', error=error)

        try:
            r = float(request.form['rate'])
            if r < 0:
                raise ValueError("Interest rate cannot be negative.")
        except ValueError:
            error = "Invalid interest rate. Please enter a non-negative number."
            return render_template('index.html', error=error)

        option_type = request.form['option_type']
        option_type_full = "Call" if option_type == "C" else "Put"

        # Fetch current stock price using yfinance
        try:
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period="1d")
            if hist.empty:
                error = "No data found for the given ticker symbol."
                return render_template('index.html', error=error)
            S = hist['Close'][0]
        except Exception as e:
            error = f"Error fetching stock price: {e}"
            return render_template('index.html', error=error)

        # Calculate time to expiration in years
        T = T_days / 365

        # Fetch historical stock data for volatility calculation
        try:
            hist = stock.history(period="1y")
            if hist.empty:
                raise ValueError("No historical data available for volatility calculation.")
            hist['Log Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            volatility = hist['Log Returns'].std() * np.sqrt(252)  # Annualize the volatility
            sigma = volatility
        except Exception as e:
            error = f"Error fetching historical data for volatility calculation: {e}"
            return render_template('index.html', error=error)

        # Calculate option price
        price = blackScholes(r, S, K, T, sigma, type=option_type)
        if price is not None:
            price = round(price, 2)
            return render_template('result.html', price=price, option_type_full=option_type_full)
        else:
            error = "Option price could not be calculated due to input errors."
            return render_template('index.html', error=error)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
