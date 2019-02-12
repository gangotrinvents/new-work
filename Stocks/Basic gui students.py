import tkinter as tk                # python 3
from tkinter import font  as tkfont
from tkinter import ttk
from tkinter import *
import numpy as np
import pandas as pd
from sklearn import linear_model
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import quandl

import os
import sys

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
from dateutil.parser import parse

count=0
count_ss=0
count_cs=0
count_dr=0
count_pt=0
count_al=0
beta_count=0
count_show_folio=0
aplha_count=0
#Decent colors
#1a3333 (dark more greenish navy blue)
def symbol_to_path(symbol,base_dir='C:/Users/Prince/Desktop/Stocks/'):
    return os.path.join(base_dir,(symbol+'.csv'))

def get_data(symbols, start_date,end_date):
    # Create an empty dataframe
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if ('SPY' not in symbols):
        symbols.insert(0, 'SPY')
    # Read data into temporary dataframe
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol),
                              index_col='Date', parse_dates=True,
                              usecols=['Date', 'Adj Close'], na_values=['nan'])
        df = df.join(df_temp, how='inner')
        df = df.rename(columns={'Adj Close': symbol})
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])
    #     print(df)
    return df

def normalize_data(df):
    return df / df.ix[0, :]

def slice_data(df1, start_date='2018-03-18', end_date='2012-03-18', company='SPY'):
    stock_names = []
    no_of_stocks = 1
    for x in range(no_of_stocks):
        stock_names.append(company)

    df_sliced = df1.ix[start_date:end_date, stock_names]

    return df_sliced

def compute_daily_returns(df):
    daily_returns=(df/df.shift(1)) - 1
    return daily_returns

def get_rolling(df,w=20):
    rmean= df.rolling(window=w,center=False).mean()
    rstddev= df.rolling(window=w,center=False).std()
    return rmean,rstddev

def get_ballinger_bands(rm,rstd):
    upper_band=rm+(2*rstd)
    lower_band=rm-(2*rstd)
    return upper_band,lower_band

def simple_mov_avg(df):
    rmean,rstddev = get_rolling(df,20)
    sma = (df/rmean)- 1
    return sma

def compute_ballinger_bands(df):
    rmean,rstddev = get_rolling(df,20)
#     sma = simple_mov_avg(df)
    print("std deviation: \n",rstddev)
    stddev = rmean + (2*rstddev)
    bbands= ((df-rmean) / (stddev))*100
    return bbands

def pred_date(no,my_num):
    for i in range(1,(no+1)):
        if(i%7 == 0):
            my_num = my_num-2
        if(i%29 == 0):
            my_num-=1
    return my_num


# bbands = compute_ballinger_bands(df_sliced)
# bbands = bbands.fillna(0)
# print("ballinger bands: \n", bbands)
#
# for date, band in zip(bbands.index, bbands.SPY):
#     if (band > 1):
#         print("Date: ", date.strftime("%d-%m-%y"), "band: ", band)

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Roboto', size=18, weight="bold")
        self.general_font = tkfont.Font(family='Roboto', size=12, weight="bold", )
        self.roboto20 = tkfont.Font(family='Roboto', size=15, weight='bold')

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        # for F in (StartPage, PageOne, PageTwo):
        #     page_name = F.__name__
        #     # print("Name is: ",page_name,",F is:",F,"page_name is of the type: ",type(page_name))
        #     frame = F(parent=container, controller=self)
        #     self.frames[page_name] = frame
        #     # put all of the pages in the same location;
        #     # the one on the top of the stacking order
        #     # will be the one that is visible.
        #     frame.grid(row=0, column=0, sticky="nsew")

        self.frames["WelcomePage"] = WelcomePage(parent=container, controller=self)
        self.frames["ActiveTrading"] = ActiveTrading(parent=container, controller=self)
        self.frames["PassiveTrading"] = PassiveTrading(parent=container, controller=self)
        self.frames["ShowStocks"] = ShowStocks(parent=container, controller=self)
        self.frames["CompareStocks"] = CompareStocks(parent=container, controller=self)
        self.frames["DailyReturns"] = DailyReturns(parent=container, controller=self)
        self.frames["Beta"] = Beta(parent=container, controller=self)
        self.frames["Portfolio"] = Portfolio(parent=container, controller=self)
        self.frames["Alpha"] = Alpha(parent=container, controller=self)

        self.frames["WelcomePage"].grid(row=0, column=0, sticky="nsew")
        self.frames["ActiveTrading"].grid(row=0, column=0, sticky="nsew")
        self.frames["PassiveTrading"].grid(row=0, column=0, sticky="nsew")
        self.frames["ShowStocks"].grid(row=0, column=0, sticky="nsew")
        self.frames["CompareStocks"].grid(row=0, column=0, sticky="nsew")
        self.frames["DailyReturns"].grid(row=0, column=0, sticky="nsew")
        self.frames["Beta"].grid(row=0, column=0, sticky="nsew")
        self.frames["Portfolio"].grid(row=0, column=0, sticky="nsew")
        self.frames["Alpha"].grid(row=0, column=0, sticky="nsew")
        self.show_frame("WelcomePage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class WelcomePage(tk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        filename = PhotoImage(file="image2.png")
        background_label = Label(self,image=filename)
        background_label.image = filename
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        label = tk.Label(background_label, text="Stock Market Trading Application",font=('roboto',30,'bold'),bg='#008080',fg='#102933')
        label.pack(side="top", fill="x", pady=10)
        # label.font= ('roboto',18)
        button1 = tk.Button(background_label, text=" Active Trading ",width=15,bg="#102933",fg="white",
                             command=lambda: controller.show_frame("ActiveTrading"))
        button2 = tk.Button(background_label, text=" Passive Trading",width=15,bg="#102933",fg="white",
                             command=lambda: controller.show_frame("PassiveTrading"))

        label1 = tk.Label(background_label,text="To first Analyze the situation in stock market,\n you can choose options from below.",
                          font=controller.general_font,bg='#67e7de',fg='#102933')
        button3 = tk.Button(background_label, text="Graph for Stocks ",bg="#102933",fg="white",width=15,
                             command=lambda: controller.show_frame("ShowStocks"))

        button4 = tk.Button(background_label, text="Stock comparison\nw.r.t. Market ",bg="#102933",fg="white",width=15,
                             command=lambda: controller.show_frame("CompareStocks"))

        button1.pack()
        button2.pack()
        label1.pack(side="top", fill="x", pady=10)
        button3.pack()
        button4.pack()
        # label_fake=tk.Label(background_label,text="")
        # label_fake.grid(row=0,column=0, rowspan=2,columnspan=10,sticky='NSEW')

        # label.grid(row=0,column=3, rowspan=2,columnspan=9,    sticky='NSEW',padx=0,pady=60)
        # button1.grid(row=2, column=0, rowspan=2,columnspan=3, sticky='NSEW',padx=300,pady=0)
        # button2.grid(row=2, column=3, rowspan=2,columnspan=3, sticky='NSEW',padx=0,pady=0)
        # label1.grid(row=4, column=0, rowspan=1,columnspan=5,  sticky='NSEW',padx=120,pady=0)
        # button3.grid(row=5, column=0, rowspan=2,columnspan=2, sticky='NSEW',padx=150,pady=0,ipadx="7",ipady="2")
        # button4.grid(row=5, column=2, rowspan=2,columnspan=2, sticky='NSEW',padx=0,pady=0)


class ActiveTrading(tk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        filename = PhotoImage(file="image1.png")
        bg_label = Label(self, image=filename)
        bg_label.image = filename
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        label = tk.Label(bg_label, text="Welcome to active trading.\n "
                                    "In this trading we use daily data to compute statistics \n"
                                    "and provide you with a prediction of today's closing price.",
                                    background = '#154360',fg='white', font=controller.title_font, justify="center")
        label.pack()#side="top", fill="x",padx=300,pady=10)
        button_port=tk.Button(bg_label, text="Portfolio ",background = '#154360',fg='white',
                               command=lambda: controller.show_frame("Portfolio"))
        button_port.pack()
        label1 = tk.Label(bg_label, text="To know the statistics of a stock before\nselecting a Stock for your Portfolio\nThe following options below can help", background = '#154360',fg='white', font=controller.general_font, justify="center")
        label1.pack()#side="top", fill="x", padx=300, pady=10)
        button1 = tk.Button(bg_label, text="Daily Returns ",background = '#154360',fg='white',
                             command=lambda: controller.show_frame("DailyReturns"))
        button1.pack()

        # img = PhotoImage(file="capm_image.png", height=19, width=198)
        # # btn = Button(root, image=img)
        # button_capm = ttk.Button(self, image=img,
        #                          command=lambda: controller.show_frame("Beta"))
        # button_capm.image = img
        # # button_capm=ttk.Button(self, text="CAPM",
        # #                    command=lambda: controller.show_frame("Beta"))
        # button_capm.pack()

        button_beta = tk.Button(bg_label, text="Beta", background = '#154360',fg='white',
                                 command=lambda: controller.show_frame("Beta"))
        button_beta.pack()
        button_alpha = tk.Button(bg_label, text="Alpha", background = '#154360',fg='white',
                                 command=lambda: controller.show_frame("Alpha"))
        button_alpha.pack()

        button_home = tk.Button(bg_label, text="Go back to home page", background = '#154360',fg='white',
                            command=lambda: controller.show_frame("WelcomePage"))
        button_home.pack()

class Beta(tk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        left_frame = ttk.Frame(self)
        left_frame.pack(padx=100, pady=10, side=LEFT)
        right_frame = ttk.Frame(self)
        right_frame.pack(padx=50, pady=10, side=LEFT)

        label1 = ttk.Label(left_frame, text=" Beta or slope w.r.t. market", font=controller.title_font,justify="center")
        label1.pack(side="top", fill="x", padx=10, pady=10)

        label2 = ttk.Label(left_frame, text="Select the stocks below.", font=controller.general_font,justify="center")
        label2.pack(side="top", fill="x", padx=10, pady=10)

        res = Label(left_frame, text="", font=("arial", 1))
        res.pack()

        bottom = ttk.Label(right_frame)
        def plot_data(df_spy_company, company,lastend,p):
            global count,bottom
            if (count > 0):
                bottom.destroy()
                count=0

            bottom = ttk.Label(right_frame)
            bottom.grid(column=0, row=1)

            f1 = Figure(figsize=(8, 3.5), dpi=100)
            axes = f1.add_subplot(111)

            axes.scatter(x=df_spy_company['SPY'], y=df_spy_company[company])
            axes.plot(lastend,p,color='green', linestyle='dashed', linewidth= 3)

            axes.set_xlabel('SPY')
            axes.set_ylabel(company)
            axes.set_title("Scatterplot between SPY and "+company+" along with the slope")
            axes.legend(loc=9)
            canvas = FigureCanvasTkAgg(f1, master=bottom)
            canvas.draw()  # canvas.show()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            # button2 = ttk.Button(right_frame, text="Go back to home page ",
            #                      command=lambda: controller.show_frame("WelcomePage"))
            button3 = ttk.Button(bottom, text="Clear ",
                                 command=lambda: bottom.destroy())
            button3.pack()
            # button2.pack()
            count += 1

        def fake_predict(x,m,c):
            l=[]
            for i in x:
                l.append(i*m+c)
            return l

        def calculate_beta(df_x, df_y, company, df_spy_company):
            global beta_count,res
            if(beta_count>0):
                res.pack_forget()
                beta_count=0
            res = Label(left_frame, text="Beta= ", font=("arial", 10, "bold"),justify="center")
            res.pack()
            mini=df_spy_company['SPY'].min()
            maxi= df_spy_company['SPY'].max()

            m = 0
            c = 0
            df_x = df_x[1:]
            df_y = df_y[1:]
            x=df_x.SPY.values
            y=df_y[company].values
            print("DAILY RETURNS OF SLICED SPY=\n",x)
            print("DAILY RETURNS OF SLICED",company,"=\n",y)

            m = (((np.mean(x) * np.mean(y)) - np.mean(x * y)) / ((np.mean(x) ** 2) - (np.mean(x ** 2))))
            c = (np.mean(y) - (m * np.mean(x)))
            # print(m)

            m_str=str(m)
            res.config(text="Beta = "+m_str)
            p = fake_predict((maxi,mini),m,c)
            # df_x, df_y
            beta_count += 1
            lastend = [maxi,mini]
            plot_data(df_spy_company, company, lastend, p)

            return m

        OPTIONS = ["Select Company ", "Google", "Apple", "IBM", "Amazon", "Netflix", "Facebook","Tesla"]  # etc

        variable = tk.StringVar(left_frame)
        variable.set(OPTIONS[0])  # default value

        COMPANY_LIST = {"Google": "GOOG",
                        "Apple": "AAPL",
                        "IBM": "IBM",
                        "Amazon": "AMZN",
                        "Netflix": "NFLX",
                        "Facebook": "FB",
                        "Tesla": "TSLA"
                        }

        def option_selection_menu(event):
            company = str(COMPANY_LIST.get(variable.get(), "GOOG"))
            start_date = '2018-01-01'
            end_date = '2018-05-18'

            # df_spy_beta = get_data([], start_date, end_date)
            df_company_SPY = get_data([company], start_date, end_date)
            df_daily=compute_daily_returns(df_company_SPY)
            df_spy_beta=slice_data(df_company_SPY, start_date, end_date,'SPY')
            spy_daily=compute_daily_returns(df_spy_beta)
            df_sliced_company_beta = slice_data(df_company_SPY, start_date, end_date, company)
            company_daily=compute_daily_returns(df_sliced_company_beta)
            calculate_beta(spy_daily,company_daily,company,df_daily)
            # plot_data(df_daily, company)
            # alpha(df_daily)
            # print(df_daily)
            return

        w = ttk.OptionMenu(left_frame, variable, *OPTIONS, command=option_selection_menu)
        w.pack()

        button2 = ttk.Button(left_frame, text="Go back to Active trading page",
                             command=lambda: controller.show_frame("ActiveTrading"))
        button2.pack()
        # ret_day = calculate_beta(df_spy, df_company) * ret_market(df_spy) + alpha(df_company)

class Alpha(tk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        left_frame = ttk.Frame(self)
        left_frame.pack(padx=100, pady=10, side=LEFT)
        right_frame = ttk.Frame(self)
        right_frame.pack(padx=50, pady=10, side=LEFT)

        label1 = ttk.Label(left_frame, text=" Prediction of today's closing price ", font=controller.title_font,justify="center")
        label1.pack(side="top", fill="x", padx=10, pady=10)

        label2 = ttk.Label(left_frame, text="Select the stocks below.", font=controller.general_font,justify="center")
        label2.pack(side="top", fill="x", padx=10, pady=10)

        res = Label(left_frame, text="", font=("arial", 1))
        # res.pack()

        bottom = ttk.Label(right_frame)
        def plot_data(X,Y, company,X_df,y_df):
            global count_al,bottom
            if (count_al > 0):
                bottom.destroy()
                count=0

            bottom = ttk.Label(right_frame)
            bottom.grid(column=0, row=1)

            f1 = Figure(figsize=(8, 3.5), dpi=100)
            axes = f1.add_subplot(111)
            axes.plot(X_df,y_df)
            axes.plot(X,Y)
            axes.set_xlabel("Days starting from 2018-03-17 to 2018-05-19")
            axes.set_ylabel("Prices")
            axes.set_title("Plot between "+company+"'s one month data with prediction")
            axes.legend(company,loc=9)
            canvas = FigureCanvasTkAgg(f1, master=bottom)
            canvas.draw()  # canvas.show()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            button3 = ttk.Button(bottom, text="Clear ",
                                 command=lambda: bottom.destroy())
            button3.pack()
            count_al += 1

        def preserve(X_preserve):
            return X_preserve

        def calculate_alpha(df_company, company,start_date,end_date):
            global count_al,res
            if(count_al>0):
                res.pack_forget()
                beta_count = 0
            res = Label(left_frame, text="Predicted value= ", font=("arial", 10, "bold"),justify="center")
            res.pack()
            # options1 = []
            # variable = tk.StringVar(left_frame)
            # variable.set(options[1])
            # w = ttk.OptionMenu(left_frame, variable, *OPTIONS, command=option_selection_menu)
            # w.pack()
            date_vals = df_company.index

            date_arr = np.array([])

            for date in (date_vals):
                date_arr = np.append(date_arr, date.strftime("%d-%m-%y"))
            no_of_days = len(date_arr)
            d = np.array([])
            # start_date = date_vals[0]
            # end_date = date_vals[-1]
            for i in range(1, (no_of_days + 1)):
                d = np.append(d, i)

            X = d[:]
            y = df_company[company].values
            X_preserve= preserve(X)
            X = X.reshape(-1, 1)
            l_model = linear_model.LinearRegression()
            l_model.fit(X, y)
            # p=(YYYY-MM-DD)
            p = "2018-05-19"
            pred_added = (parse(p) - parse(end_date)).days
            print("Before : ", pred_added)
            #
            pred_trading_days = pred_date(pred_added, pred_added)
            # print("After : ", pred_trading_days)
            # print(pred_trading_days)
            pred = no_of_days + pred_trading_days
            # print(pred)
            pred_value_1 = l_model.predict([[1]])
            pred_value_2 = l_model.predict([[pred]])
            # print("Predicted value Adj Close: ", pred_value_2)
            pred_str = str(pred_value_2)
            # m_str=str(m)
            res.config(text="Predicted value = "+pred_str)
            # df_x, df_y
            # alpha_count += 1
            plot_data([1, pred], [pred_value_1, pred_value_2],company,X_preserve,y)
            # plot_data(df_company, company, lastend, p)
            count_al+=1
            return

        OPTIONS = ["Select Company ", "Google", "Apple", "IBM", "Amazon", "Netflix", "Facebook","Tesla"]  # etc

        variable = tk.StringVar(left_frame)
        variable.set(OPTIONS[0])  # default value

        COMPANY_LIST = {"Google": "GOOG",
                        "Apple": "AAPL",
                        "IBM": "IBM",
                        "Amazon": "AMZN",
                        "Netflix": "NFLX",
                        "Facebook": "FB",
                        "Tesla": "TSLA"
                        }

        def option_selection_menu(event):
            # start = '2017-05-18'
            # end = '2018-05-18'
            company = str(COMPANY_LIST.get(variable.get(), "GOOG"))
            start_date = '2018-03-17'
            end_date = '2018-05-18'


            df_company_SPY = get_data([company], start_date, end_date)
            df_sliced_company = slice_data(df_company_SPY, start_date, end_date, company)

            calculate_alpha(df_sliced_company,company,start_date,end_date)
            return

        w = ttk.OptionMenu(left_frame, variable, *OPTIONS, command=option_selection_menu)
        w.pack()

        button2 = ttk.Button(left_frame, text="Go back to Active trading page",
                             command=lambda: controller.show_frame("ActiveTrading"))
        button2.pack()
        # ret_day = calculate_beta(df_spy, df_company) * ret_market(df_spy) + alpha(df_company)



folio_set=set()
folio_list=[]
class Portfolio(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        left_frame = ttk.Frame(self)
        left_frame.place(x=10, y=100)
        # left_frame.grid(row=0, column=0, padx=8, pady=100)
        right_frame = ttk.Frame(self)
        right_frame.place(x=500,y=100)
        # right_frame.grid(row=0, column=1, padx=20, pady=100)

        label = ttk.Label(left_frame, text="This section is to make your own Portfolio ", font=controller.general_font)
        label.pack(side="top", fill="x",padx=150, pady=10)

        label_add = ttk.Label(left_frame, text="Add stocks to the portfolio below: ", font=controller.general_font,
                          justify="center")
        label_add.pack(side="top", fill="x", padx=150, pady=10)
        label_list = ttk.Label(left_frame)

        def add_stock_to_list(comp_str):
            """pressed button's value is inserted into the end of the text area"""
            global folio_set
            folio_set.add(comp_str)
            return

        def show_folio():
            global folio_set,folio_list,count_show_folio,label_list
            if(count_show_folio>0):
                label_list.pack_forget()
                count_show_folio=0
            folio_list=list(folio_set)
            print()
            folio_str=', '.join(folio_list)
            label_list=ttk.Label(left_frame, text="My Stocks:  ", font=("arial", 10 ,"bold"),justify="center")
            label_list.config(text="My Stocks: "+folio_str )
            label_list.pack()
            count_show_folio += 1
            return

        def clear_folio():
            global folio_list,folio_set,count_show_folio
            folio_list[:]=[]
            folio_set.clear()
            # count_show_folio
            show_folio()
            return

        def plot_data(df_comp):
            global count_cs,bottom
            if(count_cs>0):
                bottom.destroy()
                count_ss=0

            bottom = ttk.Label(right_frame)
            bottom.grid(column=0, row=1)

            f1 = Figure(figsize=(8, 4), dpi=100)
            axes = f1.add_subplot(111)
            axes.plot(df_comp)
            axes.set_xlabel('Trading Days -->')
            axes.set_ylabel('Normalized Price -->')
            axes.set_title("Our portfolio in comparison to market ")
            label=['SPY','Portfolio']
            axes.legend(label,loc=9)

            canvas = FigureCanvasTkAgg(f1, master=bottom)
            canvas.draw()  # canvas.show()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
            canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            button3=ttk.Button(bottom, text="Clear ",
                                command=lambda: bottom.destroy())
            button3.pack()
            count_cs+=1

        def get_data_portfolio():
            global folio_list

            start_date = '2018-01-01'
            end_date = '2018-05-18'
            length_folio = len(folio_list)
            if(length_folio==0):
                return
            folio_list1 = []
            weight = 1/length_folio
            for x in folio_list:
                folio_list1.append(x)
            print(folio_list1)
            df = get_data(folio_list1, start_date, end_date)
            df_SPY = df.ix[start_date:end_date, ['SPY']]
            print(folio_list)
            df_sliced_company = df.ix[start_date:end_date, folio_list]
            df_normalized_company = normalize_data(df_sliced_company)
            df_normalized_SPY = normalize_data(df_SPY)
            print("##################### NORMALIZED SPY #################### \n",df_normalized_SPY)
            df_normalized_company[df_normalized_company.select_dtypes(include=['number']).columns] *= weight
            print("##################### NORMALIZED COMP #################### \n", df_normalized_company)
            s_portfolio = df_normalized_company.sum(axis=1)
            df_portfolio = s_portfolio.to_frame()
            df_portfolio1 = df_portfolio
            df_portfolio = pd.DataFrame(df_portfolio, columns=['nn'])
            df_portfolio = df_portfolio1.join(df_portfolio, how='inner')
            df_portfolio = df_portfolio.rename(columns={0: 'Portfolio'})
            df_portfolio = df_portfolio.ix[start_date: end_date, ['Portfolio']]
            df_portfolio = df_normalized_SPY.join(df_portfolio, how='inner')
            print("##################### NORMALIZED PORTFOLIO #################### \n",df_portfolio)
            plot_data(df_portfolio)
            # alpha(df_daily)
            # df_portfolio = df_normalized_company['GOOG']+df_normalized_company['AAPL']+df_normalized_company['AMZN']
            # print("########################################### summated ######################################\n",
            #       s_portfolio)
            return

        ttk.Button(left_frame, text="Google", width=8, command=lambda: add_stock_to_list('GOOG')).pack()
        ttk.Button(left_frame, text="Apple", width=8, command=lambda: add_stock_to_list('AAPL')).pack()
        ttk.Button(left_frame, text="IBM", width=8, command=lambda: add_stock_to_list('IBM')).pack()
        ttk.Button(left_frame, text="Amazon", width=8, command=lambda: add_stock_to_list('AMZN')).pack()
        ttk.Button(left_frame, text="Netflix", width=8, command=lambda: add_stock_to_list('NFLX')).pack()
        ttk.Button(left_frame, text="Facebook", width=8, command=lambda: add_stock_to_list('FB')).pack()
        ttk.Button(left_frame, text="Tesla", width=8, command=lambda: add_stock_to_list('TSLA')).pack()
        ttk.Button(left_frame, text="Show my stocks", width=15, command=show_folio).pack()
        ttk.Button(left_frame, text="Show portfolio vs Market for equal Allocations ", width=30, command=get_data_portfolio).pack()
        ttk.Button(left_frame, text="Clear stocks from Portfolio", width=30, command=clear_folio).pack()

        button2 = ttk.Button(left_frame, text="Go back to Active trading page",
                             command=lambda: controller.show_frame("ActiveTrading"))
        button2.pack()

        # def assist_beta(*x):
        #     l = []
        #     for i in x:
        #         l.append(i * m + c)
        #     return l

        button = ttk.Button(left_frame, text="Go back to home page",
                            command=lambda: controller.show_frame("WelcomePage"))
        button.pack()

class PassiveTrading(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        filename = PhotoImage(file="image3.png")
        bg_label = Label(self, image=filename)
        bg_label.image = filename
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        left_frame = Frame(bg_label, bg="#58acac")
        # top_frame.place(x=400, y=100)
        left_frame.grid(row=0, column=0, padx=50, pady=100)
        right_frame = Frame(bg_label, bg="#58acac")
        # bottom_frame.place(x=400,y=100)
        right_frame.grid(row=0, column=1, padx=20, pady=100)

        label = tk.Label(left_frame, text="Long term Investment",bg="#58acac",fg="#1a3333", font=controller.general_font,justify="center")
        label.pack(side="top", fill="x",padx=150, pady=10)
        button_pred = tk.Button(left_frame, text="Predict prices for the next year",bg="#154360")
                           # command=lambda: controller.show_frame("WelcomePage"))
        button_pred.pack()

        OPTIONS = ["Select Company", "SPY", "Google", "Apple", "IBM", "Amazon", "Netflix", "Facebook", "Tesla"]  # etc

        COMPANY_LIST = {"SPY": "SPY",
                        "Google": "GOOG",
                        "Apple": "AAPL",
                        "IBM": "IBM",
                        "Amazon": "AMZN",
                        "Netflix": "NFLX",
                        "Facebook": "FB",
                        "Tesla": "TSLA"
                        }

        variable = tk.StringVar(left_frame)
        variable.set(OPTIONS[0])  # default value
        bottom=ttk.Label(right_frame)
        def plot_data(df):
            global count_pt,bottom
            if(count_pt>0):
                bottom.destroy()

            bottom = ttk.Label(right_frame)
            bottom.grid(column=0, row=1)

            f1 = Figure(figsize=(8, 4), dpi=100)
            axes = f1.add_subplot(111)
            axes.plot(df)
            axes.set_xlabel('Trading Days since 2012-->')
            axes.set_ylabel('Price in dollars-->')
            axes.set_title("Stock Pricing upto the present day ")

            axes.legend(loc='center')
            canvas = FigureCanvasTkAgg(f1, master=bottom)
            canvas.draw()  # canvas.show()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
            # button2= ttk.Button(right_frame, text="Go back to home page ",
            #                     command=lambda: controller.show_frame("WelcomePage"))
            label_showing=ttk.Label(bottom,text="Showing data from the year 2015-2018 ", font=controller.general_font,justify='center')

            button3=ttk.Button(bottom, text="Clear ",
                                command=lambda: bottom.destroy())
            button3.pack()
            label_showing.pack()
            # button2.pack()
            count_pt+=1

        def option_selection_menu(event):
            company = str(COMPANY_LIST.get(variable.get(), "SPY"))
            start_date = '2015-01-01'
            end_date = '2018-01-30'
            df = get_data([company], start_date, end_date)
            df_sliced = slice_data(df, start_date, end_date, company)
            rmean, rstddev = get_rolling(df_sliced, 20)
            plot_data(rmean)

            return


        w = ttk.OptionMenu(left_frame, variable, *OPTIONS, command=option_selection_menu)#,bg="#154360",fg="white")
        w.pack()
        button = tk.Button(left_frame, text="Go back to home page",bg="#154360",fg="white",
                                   command=lambda: controller.show_frame("WelcomePage"))
        button.pack()

class ShowStocks(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        left_frame=Frame(self,bg="gainsboro")
        # top_frame.place(x=400, y=100)
        left_frame.grid(row=0,column=0,padx=100,pady=100)
        right_frame=Frame(self)
        # bottom_frame.place(x=400,y=100)
        right_frame.grid(row=0,column=1,padx=20,pady=100)
        label = ttk.Label(left_frame, text="Select the stocks below.", font=controller.title_font)
        label.pack(side="top", fill="x",padx=10, pady=10)
        button = ttk.Button(left_frame, text="Go back to home page",
                            command=lambda: controller.show_frame("WelcomePage"))
        button.pack()

        OPTIONS = ["Select Company","SPY","Google","Apple","IBM","Amazon","Netflix","Facebook","Tesla"]  # etc

        COMPANY_LIST = {"SPY": "SPY",
                        "Google": "GOOG",
                        "Apple": "AAPL",
                        "IBM": "IBM",
                        "Amazon":"AMZN",
                        "Netflix":"NFLX",
                        "Facebook":"FB",
                        "Tesla":"TSLA"
                        }
        variable = tk.StringVar(left_frame)
        variable.set(OPTIONS[0])  # default value
        bottom = ttk.Label(right_frame)

        def plot_data(df):
            global count_ss,bottom
            if(count_ss>0):
                bottom.pack_forget()
                count_ss=0

            bottom = ttk.Label(right_frame)
            bottom.pack()

            f1 = Figure(figsize=(8, 4), dpi=100)
            axes = f1.add_subplot(111)
            axes.plot(df)
            axes.set_xlabel('Trading Days since 2012-->')
            axes.set_ylabel('Price -->')
            axes.set_title("Stock Pricing upto the present day ")
            axes.legend(loc='upper left')
            canvas = FigureCanvasTkAgg(f1, master=bottom)
            canvas.draw()  # canvas.show()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            label_showing=ttk.Label(bottom,text="Showing data from the year 2017-2018 ", font=controller.general_font,justify='center')

            button3=ttk.Button(bottom, text="Clear ",
                               command=lambda: bottom.destroy())
            button3.pack()
            label_showing.pack()
            count_ss+=1

        def option_selection_menu(event):
            start_date = '2012-03-01'
            end_date = '2018-03-01'
            company = str(COMPANY_LIST.get(variable.get(),"SPY"))
            start_date = '2017-03-18'
            end_date = '2018-03-18'
            df = get_data([company], start_date, end_date)
            df_sliced= slice_data(df,start_date,end_date,company)
            plot_data(df_sliced)
            return

        w = ttk.OptionMenu(left_frame, variable, *OPTIONS, command=option_selection_menu)
        w.pack()

class CompareStocks(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        left_frame=ttk.Frame(self)
        # left_frame.place(x=400, y=100)
        left_frame.grid(row=0,column=0,padx=100,pady=100)
        right_frame=ttk.Frame(self)
        # right_frame.place(x=400,y=100)
        right_frame.grid(row=0,column=1,padx=20,pady=100)
        label = ttk.Label(left_frame, text="Select the stocks below.", font=controller.title_font,justify='center')
        label.pack(side="top", fill="x",padx=10, pady=10)
        button = ttk.Button(left_frame, text="Go back to home page",
                           command=lambda: controller.show_frame("WelcomePage"))
        button.pack()


        OPTIONS = ["Select Company to compare ","Google","Apple","IBM","Amazon","Netflix","Facebook","Tesla"]  # etc

        COMPANY_LIST = {"Google": "GOOG",
                        "Apple": "AAPL",
                        "IBM": "IBM",
                        "Amazon":"AMZN",
                        "Netflix":"NFLX",
                        "Facebook":"FB",
                        "Tesla":"TSLA"
                        }
        variable = tk.StringVar(left_frame)
        variable.set(OPTIONS[0])  # default value
        bottom = ttk.Label(right_frame)

        def plot_data(df_comp,company):
            global count_cs,bottom
            if(count_cs>0):
                bottom.destroy()
                count_ss=0

            bottom = ttk.Label(right_frame)
            bottom.grid(column=0, row=1)

            f1 = Figure(figsize=(8, 4), dpi=100)
            axes = f1.add_subplot(111)
            axes.plot(df_comp)
            axes.set_xlabel('Trading Days since 2012-->')
            axes.set_ylabel('Price -->')
            axes.set_title("Stock Pricing upto the present day ")
            label=['SPY',company]
            axes.legend(label,loc=9)
            canvas = FigureCanvasTkAgg(f1, master=bottom)

            canvas.draw()  # canvas.show()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
            # button2= ttk.Button(right_frame, text="Go back to home page ",
            #                     command=lambda: controller.show_frame("WelcomePage"))
            button3=ttk.Button(bottom, text="Clear ",
                                command=lambda: bottom.destroy())
            button3.pack()
            # button2.pack()
            count_cs+=1

        def option_selection_menu(event):
            start_date = '2012-03-01'
            end_date = '2018-03-01'
            company = str(COMPANY_LIST.get(variable.get(),"GOOG"))
            start_date = '2012-03-18'
            end_date = '2018-03-18'
            df_stock = get_data([company], start_date, end_date)
            # df_stock_sliced= slice_data(df,start_date,end_date,company)

            df_norm = normalize_data(df_stock)
            plot_data(df_norm,company)
            return

        w = ttk.OptionMenu(left_frame, variable, *OPTIONS, command=option_selection_menu)
        w.pack()

class DailyReturns(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        left_frame=ttk.Frame(self)
        # top_frame.place(x=400, y=100)
        left_frame.pack(padx=100 ,side=LEFT)
        right_frame=ttk.Frame(self)
        # bottom_frame.place(x=400,y=100)
        right_frame.pack(padx=10, side=LEFT)
        label = ttk.Label(left_frame, text="Daily Returns", font=controller.title_font,justify='center')
        label.pack(side="top", fill="x",padx=10, pady=10)
        button = ttk.Button(left_frame, text="Go back to home page",
                            command=lambda: controller.show_frame("WelcomePage"))
        button.pack()


        OPTIONS = ["Select Company ", "Google", "Apple", "IBM", "Amazon", "Netflix", "Facebook",
                   "Tesla"]  # etc

        COMPANY_LIST = {"Google": "GOOG",
                        "Apple": "AAPL",
                        "IBM": "IBM",
                        "Amazon": "AMZN",
                        "Netflix": "NFLX",
                        "Facebook": "FB",
                        "Tesla": "TSLA"
                        }
        variable = tk.StringVar(left_frame)
        variable.set(OPTIONS[0])  # default value
        bottom = ttk.Label(right_frame)

        def plot_data(df_comp):
            global count_dr,bottom
            if (count_dr > 0):
                bottom.destroy()
                count_dr=0

            bottom = ttk.Label(right_frame)
            bottom.grid(column=0, row=1)

            # bottom.pack(padx=5, pady=10)
            # w.pack(padx=5, pady=10, side=LEFT)
            f1 = Figure(figsize=(8, 4), dpi=100)
            axes = f1.add_subplot(111)
            axes.plot(df_comp)
            axes.set_xlabel('Trading Days since 2012-->')
            axes.set_ylabel('Price -->')
            axes.set_title("Stock Pricing upto the present day ")
            axes.legend(loc=9)
            canvas = FigureCanvasTkAgg(f1, master=bottom)

            canvas.draw()  # canvas.show()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
            # button2 = ttk.Button(right_frame, text="Go back to home page ",
            #                      command=lambda: controller.show_frame("WelcomePage"))
            button3 = ttk.Button(bottom, text="Clear ",
                                 command=lambda: bottom.destroy())
            button3.pack()
            # button2.pack()
            count_dr += 1

        def option_selection_menu(event):
            start = '2018-04-01'
            end = '2018-05-18'
            company = str(COMPANY_LIST.get(variable.get(), "GOOG"))
            start_date = '2018-04-01'
            end_date = '2018-05-18'
            df_stock = get_data([company], start, end)
            # df_stock_sliced= slice_data(df_stock,start_date,end_date,company)

            df_daily = compute_daily_returns(df_stock)
            plot_data(df_daily)
            print(df_daily)
            return

        w = ttk.OptionMenu(left_frame, variable, *OPTIONS, command=option_selection_menu)
        w.pack()

if __name__ == "__main__":
    app = SampleApp()
    app.state('zoomed')
    app.mainloop()

################################################### End ################################################################
# rmean, rstddev = get_rolling(df_daily,5)
# rmean.plot(label="Rolling mean", ax=axes)
#
# upperband,lowerband= get_ballinger_bands(rmean,rstddev)
# upperband.plot(label="Upper Band",ax=axes)
# lowerband.plot(label="lower Band",ax=axes)

# labels = ['Daily ret.', 'Rolling mean', 'Upper band', 'Lower band']
# axes.legend(labels,loc='upper right')


# OPTIONS = ["Select Company","SPY","Google","Apple","IBM","Amazon","Netflix","Facebook","Tesla"]  # etc
#
#  COMPANY_LIST = {"SPY": "SPY",
#                         "Google": "GOOG",
#                         "Apple": "AAPL",
#                         "IBM": "IBM",
#                         "Amazon":"AMZN",
#                         "Netflix":"NFLX",
#                         "Facebook":"FB",
#                         "Tesla":"TSLA"
#                         }
#         variable = tk.StringVar(top_frame)
#         variable.set(OPTIONS[0])  # default value
#
#
# def option_selection_menu(event):
#     company = str(COMPANY_LIST.get(variable.get(), "SPY"))
#     start_date = '2015-01-01'
#     end_date = '2018-01-30'
#     df = get_data([company], start_date, end_date)
#     df_sliced = slice_data(df, start_date, end_date, company)
#
#     plot_data(df_sliced)
#     # print(df)
#     return
#
#
# w = ttk.OptionMenu(top_frame, variable, *OPTIONS, command=option_selection_menu)
# w.pack()

