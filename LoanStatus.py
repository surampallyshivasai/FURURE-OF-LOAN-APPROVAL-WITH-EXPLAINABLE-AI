from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except Exception:
    MPLCURSORS_AVAILABLE = False
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier#importing ML classes
import shap


main = tkinter.Tk()
main.title("Future of Loan Approvals with Explainable AI") #designing main screen
main.geometry("1100x720")

# Use ttk theme for a more modern look
style = ttk.Style()
try:
    style.theme_use('clam')
except Exception:
    pass

# Hacker theme colors and fonts
HACKER_BG = '#000000'
HACKER_FG = '#00FF66'
HACKER_ACCENT = '#00CC00'
MONO_FONT = ('Consolas', 10)

# Apply some ttk style tweaks for dark theme
style.configure('.', background=HACKER_BG, foreground=HACKER_FG)
style.configure('TLabel', background=HACKER_BG, foreground=HACKER_FG, font=('Consolas', 12))
style.configure('TButton', background='#111111', foreground=HACKER_FG, font=('Consolas', 10))
style.configure('Treeview', background='#0b0b0b', foreground=HACKER_FG, fieldbackground='#0b0b0b')
style.map('TButton', background=[('active', '#222222')])

# --- Gradient button widget for modern look ---
class GradientButton(tkinter.Canvas):
    def __init__(self, master, text, command=None, width=200, height=36, from_color='#00FF66', to_color='#008844', radius=8, **kwargs):
        super().__init__(master, width=width, height=height, highlightthickness=0, bg=HACKER_BG)
        # avoid overwriting internal tkinter attributes like _w; use clearer names
        self._width = width
        self._height = height
        self._text = text
        self._cmd = command
        self._from = self._hex_to_rgb(from_color)
        self._to = self._hex_to_rgb(to_color)
        self._radius = radius
        self._text_id = None
        self.draw_gradient()
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))

    def _hex_to_rgb(self, h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb

    def draw_gradient(self):
        self.delete('all')
        steps = max(2, int(self._height))
        for i in range(steps):
            t = i / (steps - 1)
            r = int(self._from[0] * (1 - t) + self._to[0] * t)
            g = int(self._from[1] * (1 - t) + self._to[1] * t)
            b = int(self._from[2] * (1 - t) + self._to[2] * t)
            color = self._rgb_to_hex((r, g, b))
            self.create_line(0, i, self._width, i, fill=color)
        # draw text
        self._text_id = self.create_text(self._width//2, self._height//2, text=self._text, fill=HACKER_BG, font=('Consolas', 10, 'bold'))

    def _on_click(self, event):
        if callable(self._cmd):
            try:
                self._cmd()
            except Exception as e:
                log_message(f'Button error: {e}')

    def _hover(self, on):
        if on:
            self.scale('all', self._width/2, self._height/2, 1.02, 1.02)
        else:
            self.delete('all')
            self.draw_gradient()


# Initialize module-level variables (avoid top-level `global` statements)
filename = None
dataset = None
X = None
loan_status = None
loan_reject_reason = None
status_names = None
reject_names = None
label_encoder = []
scaler = None
cols = []

loan_X_train = loan_X_test = loan_y_train = loan_y_test = None
reject_X_train = reject_X_test = reject_y_train = reject_y_test = None

accuracy = []
precision = []
recall = []
fscore = []

rf = None
reject_rf = None

# UI widgets (initialized later)
metrics_tree = None
plot_frame = None
log_text = None
results_tree = None

# --- simple UI helpers (defined early so functions can call them) ---
def log_message(msg):
    try:
        # if the UI log_text exists, append there
        if log_text is not None:
            log_text.insert(END, msg + '\n')
            log_text.see('end')
        else:
            print(msg)
    except Exception:
        print(msg)

def log_clear():
    try:
        if log_text is not None:
            log_text.delete('1.0', END)
    except Exception:
        pass

def save_log():
    try:
        if log_text is None:
            log_message('No log available to save')
            return
        file = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text files','*.txt'), ('All files','*.*')], initialdir='.')
        if not file:
            return
        content = log_text.get('1.0', END)
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        log_message(f'Log saved to: {file}')
    except Exception as e:
        log_message(f'Error saving log: {e}')

def add_result_entry(index, status_text, reject_text, extras=None):
    """Insert a result row into the interactive results Treeview."""
    try:
        if results_tree is None:
            # fallback to textual log
            log_message(f"Index {index} - Status: {status_text} - Reason: {reject_text}")
            return
        tag = 'unknown'
        s_lower = str(status_text).lower()
        if 'approve' in s_lower or 'approved' in s_lower or 'acc' in s_lower or s_lower=='1':
            tag = 'approved'
        elif 'reject' in s_lower or 'rejected' in s_lower or 'refuse' in s_lower or s_lower=='0':
            tag = 'rejected'
        values = (index, status_text, reject_text)
        iid = results_tree.insert('', 'end', values=values, tags=(tag,))
        # optionally store extras in item map
        if extras is not None:
            results_tree.set(iid, 'extras', str(extras))
        # auto-scroll
        results_tree.see(iid)
    except Exception:
        log_message(f"Result: {index} - {status_text} - {reject_text}")

def save_selected_results():
    try:
        if results_tree is None:
            log_message('No results to save')
            return
        sel = results_tree.selection()
        if not sel:
            log_message('No rows selected')
            return
        file = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv'),('All','*.*')], initialdir='.')
        if not file:
            return
        rows = []
        for iid in sel:
            rows.append(results_tree.item(iid)['values'])
        # write CSV
        import csv
        with open(file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Index','Approval Status','Reject Reason'])
            for r in rows:
                writer.writerow(r)
        log_message(f'Selected results saved to: {file}')
    except Exception as e:
        log_message(f'Error saving selected results: {e}')

def clear_results():
    try:
        if results_tree is None:
            return
        for item in results_tree.get_children():
            results_tree.delete(item)
        log_message('Results cleared')
    except Exception:
        pass

def on_result_double_click(event):
    try:
        if results_tree is None:
            return
        sel = results_tree.selection()
        if not sel:
            return
        iid = sel[0]
        vals = results_tree.item(iid)['values']
        # show a small details popup
        d = Toplevel(main)
        d.title(f'Result Details - Index {vals[0]}')
        d.config(bg=HACKER_BG)
        ttk.Label(d, text=f'Index: {vals[0]}', background=HACKER_BG, foreground=HACKER_FG).pack(anchor='w', padx=8, pady=4)
        ttk.Label(d, text=f'Approval Status: {vals[1]}', background=HACKER_BG, foreground=HACKER_FG).pack(anchor='w', padx=8, pady=4)
        ttk.Label(d, text=f'Reject Reason: {vals[2]}', background=HACKER_BG, foreground=HACKER_FG, wraplength=420).pack(anchor='w', padx=8, pady=4)
        ttk.Button(d, text='Close', command=d.destroy).pack(pady=8)
    except Exception as e:
        log_message(f'Error showing details: {e}')

def clear_metrics():
    try:
        if metrics_tree is None:
            return
        for item in metrics_tree.get_children():
            metrics_tree.delete(item)
        log_message('Metrics cleared')
    except Exception:
        pass

# Loader / spinner for interesting loading animation
loader_running = False
loader_index = 0
loader_chars = ['|','/','-','\\']
spinner_label = None

def loader_step():
    global loader_index
    if not loader_running:
        return
    try:
        if spinner_label is not None:
            spinner_label.config(text=f" LOADING {loader_chars[loader_index]} ", fg=HACKER_ACCENT, bg=HACKER_BG, font=('Consolas', 11, 'bold'))
        loader_index = (loader_index + 1) % len(loader_chars)
        # schedule next frame
        main.after(120, loader_step)
    except Exception:
        pass

def loader_start():
    global loader_running, loader_index
    loader_running = True
    loader_index = 0
    loader_step()

def loader_stop():
    global loader_running
    loader_running = False
    try:
        if spinner_label is not None:
            spinner_label.config(text='')
    except Exception:
        pass



def loadDataset():
    global filename, dataset, loan_status, loan_reject_reason, status_names, reject_names
    filename = filedialog.askopenfilename(initialdir="Dataset")
    if not filename:
        return
    dataset_label.config(text=os.path.basename(filename))
    log_message(f"{filename} loaded")
    dataset = pd.read_csv(filename, nrows=20000)
    log_message(str(dataset.head()))
    loan_status = dataset['NAME_CONTRACT_STATUS']
    loan_reject_reason = dataset['CODE_REJECT_REASON']
    #visualizing loan status class labels
    status_names, status_count = np.unique(loan_status, return_counts = True)
    reject_names, reject_count = np.unique(loan_reject_reason, return_counts = True)
    loan_df = []
    for i in range(len(status_names)):
        loan_df.append([status_names[i], status_count[i]])
    loan_df = pd.DataFrame(loan_df, columns=['Loan_Status', 'Count'])
    reject_df = []
    for i in range(len(reject_names)):
        reject_df.append([reject_names[i], reject_count[i]])
    reject_df = pd.DataFrame(reject_df, columns=['Reject_Reason', 'Count'])    

    # embed the summary bar plots in the GUI
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    sns.barplot(x="Loan_Status", y='Count', data=loan_df, ax=axs[0])
    sns.barplot(x="Reject_Reason", y='Count', data=reject_df, ax=axs[1])
    axs[0].set_title("Loan Application Status")
    axs[1].set_title("Reject Reason")
    # clear previous canvas if present
    for child in plot_frame.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    # populate feature selector with dataset column names (safe if widget exists)
    try:
        feature_listbox.delete(0, 'end')
        for c in dataset.columns:
            feature_listbox.insert('end', c)
    except Exception:
        pass

def processDataset():
    log_clear()
    global dataset, scaler, label_encoder, cols, loan_status, loan_reject_reason, X
    label_encoder = []
    #converting non-numeric data to numeric values
    dataset.fillna(0, inplace = True)
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    cols = []
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
            label_encoder.append(le)
            cols.append(columns[i])
    dataset.fillna(0, inplace = True)
    loan_status = dataset['NAME_CONTRACT_STATUS']
    loan_reject_reason = dataset['CODE_REJECT_REASON']
    dataset.drop(['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON'], axis = 1,inplace=True)
    X = dataset.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)#normalize train features
    log_message(f"Processed features shape: {X.shape}")
    # enable split button
    split_btn.config(state='normal')
    # refresh feature selector to show processed feature names (targets removed)
    try:
        feature_listbox.delete(0, 'end')
        for c in dataset.columns:
            feature_listbox.insert('end', c)
    except Exception:
        pass

def splitDataset():
    log_clear()
    global X, loan_status, loan_reject_reason
    global loan_X_train, loan_X_test, loan_y_train, loan_y_test
    global reject_X_train, reject_X_test, reject_y_train, reject_y_test
    #split dataset into train and test
    loan_X_train, loan_X_test, loan_y_train, loan_y_test = train_test_split(X, loan_status, test_size = 0.2)
    reject_X_train, reject_X_test, reject_y_train, reject_y_test = train_test_split(X, loan_reject_reason, test_size = 0.2)
    log_message(f"Total records found in dataset = {X.shape[0]}")
    log_message(f"Total features found in dataset = {X.shape[1]}")
    log_message(f"80% dataset for training : {loan_X_train.shape[0]}")
    log_message(f"20% dataset for testing  : {loan_X_test.shape[0]}")
    # enable training buttons
    approval_btn.config(state='normal')
    reject_btn.config(state='normal')

def calculateMetrics(algorithm, predict, y_test, label_names):
    # compute metrics (safe to run in worker threads)
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    # prepare the confusion matrix for plotting
    conf_matrix = confusion_matrix(y_test, predict)

    # schedule GUI updates on the main thread to avoid tkinter / matplotlib threading issues
    def _update_ui():
        try:
            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            fscore.append(f)
            log_message(f"{algorithm} Accuracy  :  {a}")
            log_message(f"{algorithm} Precision : {p}")
            log_message(f"{algorithm} Recall    : {r}")
            log_message(f"{algorithm} FScore    : {f}")

            # add/update metrics table
            if metrics_tree is not None:
                metrics_tree.insert('', 'end', values=(algorithm, f"{a:.2f}", f"{p:.2f}", f"{r:.2f}", f"{f:.2f}"))

            # draw confusion matrix using matplotlib from the main thread
            try:
                plt.figure(figsize=(7, 5))
                ax = sns.heatmap(conf_matrix, xticklabels=label_names, yticklabels=label_names, annot=True, cmap="viridis", fmt="g")
                ax.set_ylim([0, len(label_names)])
                plt.title(algorithm + " Confusion matrix")
                plt.xticks(rotation=90)
                plt.ylabel('True class')
                plt.xlabel('Predicted class')
                plt.tight_layout()

                # embed confusion matrix into GUI (replace previous canvas)
                for child in plot_frame.winfo_children():
                    child.destroy()
                canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
            except Exception as e:
                log_message(f"Failed to draw confusion matrix on UI: {e}")
        except Exception as ex:
            log_message(f"Error updating metrics UI: {ex}")

    try:
        # run UI update on main thread
        main.after(0, _update_ui)
    except Exception:
        # fallback: if scheduling fails, try direct update
        _update_ui()

def aiApproval():
    log_clear()
    global loan_X_train, loan_X_test, loan_y_train, loan_y_test
    global reject_X_train, reject_X_test, reject_y_train, reject_y_test
    global accuracy, precision, recall, fscore, rf, status_names
    #define global variables to save accuracy and other metrics
    accuracy = []
    precision = []
    recall = []
    fscore = []
    rf = RandomForestClassifier()
    def job():
        try:
            progress.start()
            loader_start()
            rf.fit(loan_X_train, loan_y_train)
            predict = rf.predict(loan_X_test)
            calculateMetrics("Random Forest Loan Status", predict, loan_y_test, status_names)
        finally:
            progress.stop()
            loader_stop()
    threading.Thread(target=job, daemon=True).start()

def aiReject():
    global reject_X_train, reject_X_test, reject_y_train, reject_y_test
    global accuracy, precision, recall, fscore, rf, reject_names, reject_rf
    reject_rf = RandomForestClassifier()
    def job():
        try:
            progress.start()
            loader_start()
            reject_rf.fit(reject_X_train, reject_y_train)
            predict = reject_rf.predict(reject_X_test)
            calculateMetrics("Random Forest Loan Rejection", predict, reject_y_test, reject_names)
        finally:
            progress.stop()
            loader_stop()
    threading.Thread(target=job, daemon=True).start()

def explainAI():
    global rf
    # dynamically obtain feature names from the dataset (after drop of targets)
    try:
        feature_names = dataset.columns.tolist()
    except Exception:
        feature_names = None
    def job():
        try:
            progress.start()
            loader_start()
            # perform heavy shap calculations in worker thread
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(loan_X_test[0:200])

            # schedule the plotting / GUI embed to run on the main thread
            def _plot_shap():
                try:
                    plt.figure(figsize=(8,5))
                    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
                    fig = plt.gcf()
                    # embed into GUI
                    for child in plot_frame.winfo_children():
                        child.destroy()
                    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                    canvas.draw()
                    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                    toolbar.update()
                    canvas.get_tk_widget().pack(fill='both', expand=True)
                    # close the matplotlib figure to free memory (canvas holds a copy)
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                except Exception as e:
                    log_message(f"Error drawing SHAP UI: {e}")

            try:
                main.after(0, _plot_shap)
            except Exception:
                _plot_shap()
        finally:
            progress.stop()
            loader_stop()
    threading.Thread(target=job, daemon=True).start()
    

def predict():
    log_clear()
    global rf, reject_rf, scaler, label_encoder, cols, status_names, reject_names
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test_data = pd.read_csv(filename)
    test_data.fillna(0, inplace = True)
    for i in range(len(cols)):
        test_data[cols[i]] = pd.Series(label_encoder[i].transform(test_data[cols[i]].astype(str)))
        test_data.fillna(0, inplace = True)
    test_data.drop(['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON'], axis = 1,inplace=True)
    test = test_data.values
    X = test_data.values
    X = scaler.transform(X)
    loan_predict = rf.predict(X)
    reject_predict = reject_rf.predict(X)
    for i in range(len(loan_predict)):
        try:
            status_text = status_names[loan_predict[i]]
        except Exception:
            status_text = str(loan_predict[i])
        try:
            reject_text = reject_names[reject_predict[i]]
        except Exception:
            reject_text = str(reject_predict[i])
        # insert an interactive result entry instead of plain text
        add_result_entry(i, status_text, reject_text)
    

# ----------------- Modern Tabbed UI Layout -----------------
header = ttk.Frame(main)
header.pack(fill='x', padx=12, pady=10)

title = ttk.Label(header, text='Future of Loan Approvals with Explainable AI', font=('Segoe UI', 18, 'bold'))
title.pack(side='left')

# Main notebook with three tabs
notebook = ttk.Notebook(main)
notebook.pack(fill='both', expand=True, padx=10, pady=8)

# Dashboard tab: interactive plots and feature selector
dashboard_tab = ttk.Frame(notebook)
notebook.add(dashboard_tab, text='Dashboard')

dash_left = ttk.Frame(dashboard_tab, width=260)
dash_left.pack(side='left', fill='y', padx=8, pady=8)

dataset_label = ttk.Label(dash_left, text='No file selected', wraplength=220)
dataset_label.pack(pady=(0,8))

upload_btn = ttk.Button(dash_left, text='Upload Dataset', command=loadDataset)
upload_btn.pack(fill='x', pady=4)

process_btn = ttk.Button(dash_left, text='Preprocess Dataset', command=processDataset)
process_btn.pack(fill='x', pady=4)

split_btn = ttk.Button(dash_left, text='Split Train/Test', command=splitDataset, state='disabled')
split_btn.pack(fill='x', pady=4)

ttk.Separator(dash_left, orient='horizontal').pack(fill='x', pady=6)

ttk.Label(dash_left, text='Feature Selector', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
feature_listbox = Listbox(dash_left, selectmode='multiple', width=32, height=12)
feature_listbox.pack(pady=4)

plot_buttons = ttk.Frame(dash_left)
plot_buttons.pack(fill='x', pady=4)

def plot_selected_distribution():
    sel = feature_listbox.curselection()
    if not sel:
        log_message('No feature selected')
        return
    feat = feature_listbox.get(sel[0])
    if dataset is None:
        log_message('Dataset not loaded')
        return
    fig = plt.figure(figsize=(6,4))
    sns.histplot(dataset[feat].dropna(), kde=True, color=HACKER_ACCENT)
    plt.title(f'Distribution: {feat}', color=HACKER_FG)
    plt.gca().set_facecolor('#020202')
    for child in plot_frame.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def plot_correlation():
    if dataset is None:
        log_message('Dataset not loaded')
        return
    corr = dataset.corr()
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(corr, cmap='viridis')
    plt.title('Feature Correlation', color=HACKER_FG)
    for child in plot_frame.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def plot_3d_scatter():
    # require at least 3 numeric columns
    if dataset is None:
        log_message('Dataset not loaded')
        return
    numeric = dataset.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) < 3:
        log_message('Need at least 3 numeric features for 3D scatter')
        return
    xcol, ycol, zcol = numeric[0], numeric[1], numeric[2]
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset[xcol], dataset[ycol], dataset[zcol], c=dataset[xcol], cmap='plasma', s=12)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_zlabel(zcol)
    ax.set_facecolor('#020202')
    for child in plot_frame.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def plot_3d_surface():
    if dataset is None:
        log_message('Dataset not loaded')
        return
    numeric = dataset.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) < 3:
        log_message('Need at least 3 numeric features for 3D surface')
        return
    xcol, ycol, zcol = numeric[0], numeric[1], numeric[2]
    # create grid
    Xg = np.linspace(dataset[xcol].min(), dataset[xcol].max(), 30)
    Yg = np.linspace(dataset[ycol].min(), dataset[ycol].max(), 30)
    Xg, Yg = np.meshgrid(Xg, Yg)
    # approximate Z by nearest neighbor interpolation (simple)
    Zg = np.zeros_like(Xg)
    xi = dataset[xcol].values
    yi = dataset[ycol].values
    zi = dataset[zcol].values
    from scipy.interpolate import griddata
    Zg = griddata((xi, yi), zi, (Xg, Yg), method='linear')
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, Zg, cmap='plasma', linewidth=0, antialiased=False)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_zlabel(zcol)
    for child in plot_frame.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill='both', expand=True)

GradientButton(plot_buttons, 'Plot Distribution', command=plot_selected_distribution, width=180, height=34, from_color='#00FF66', to_color='#00CC66').pack(pady=4)
GradientButton(plot_buttons, 'Plot Correlation', command=plot_correlation, width=180, height=34, from_color='#00CC66', to_color='#009944').pack(pady=4)
GradientButton(plot_buttons, '3D Scatter', command=plot_3d_scatter, width=180, height=34, from_color='#00AAFF', to_color='#0066FF').pack(pady=4)
GradientButton(plot_buttons, '3D Surface', command=plot_3d_surface, width=180, height=34, from_color='#FF66CC', to_color='#FF3399').pack(pady=4)

center_dash = ttk.Frame(dashboard_tab)
center_dash.pack(side='left', fill='both', expand=True, padx=6, pady=8)

plot_frame = ttk.Frame(center_dash, relief='sunken')
plot_frame.pack(fill='both', expand=True)

# Models tab: training controls and metrics
models_tab = ttk.Frame(notebook)
notebook.add(models_tab, text='Models')

models_left = ttk.Frame(models_tab, width=280)
models_left.pack(side='left', fill='y', padx=8, pady=8)

approval_btn = ttk.Button(models_left, text='Train Approval Model', command=aiApproval, state='disabled')
approval_btn.pack(fill='x', pady=4)

reject_btn = ttk.Button(models_left, text='Train Reject Model', command=aiReject, state='disabled')
reject_btn.pack(fill='x', pady=4)

explain_btn = ttk.Button(models_left, text='Explainable AI (SHAP)', command=explainAI)
explain_btn.pack(fill='x', pady=4)

progress = ttk.Progressbar(models_left, mode='indeterminate')
progress.pack(fill='x', pady=6)

save_log_btn = ttk.Button(models_left, text='Save Log', command=save_log)
save_log_btn.pack(fill='x', pady=4)

# spinner label for hacker-style loading
spinner_label = Label(models_left, text='', bg=HACKER_BG, fg=HACKER_FG, font=('Consolas', 11, 'bold'))
spinner_label.pack(fill='x', pady=(6,4))

metrics_frame = ttk.Frame(models_tab)
metrics_frame.pack(side='left', fill='both', expand=True, padx=6, pady=8)

metrics_label = ttk.Label(metrics_frame, text='Model Metrics', font=('Segoe UI', 12, 'bold'))
metrics_label.pack(anchor='w')

metrics_tree = ttk.Treeview(metrics_frame, columns=('algo','acc','prec','rec','f1'), show='headings', height=8)
metrics_tree.heading('algo', text='Algorithm')
metrics_tree.heading('acc', text='Accuracy')
metrics_tree.heading('prec', text='Precision')
metrics_tree.heading('rec', text='Recall')
metrics_tree.heading('f1', text='F1')
metrics_tree.column('algo', width=160)
metrics_tree.pack(fill='x', pady=6)

clear_metrics_btn = ttk.Button(metrics_frame, text='Clear Metrics', command=clear_metrics)
clear_metrics_btn.pack(anchor='e', pady=4)

# Predict & Logs tab
predict_tab = ttk.Frame(notebook)
notebook.add(predict_tab, text='Predict / Logs')

predict_controls = ttk.Frame(predict_tab)
predict_controls.pack(fill='x', padx=8, pady=8)

predict_btn = ttk.Button(predict_controls, text='Predict (Select Test CSV)', command=predict)
predict_btn.pack(side='left', padx=6)

save_results_btn = ttk.Button(predict_controls, text='Save Selected', command=save_selected_results)
save_results_btn.pack(side='left', padx=6)

clear_results_btn = ttk.Button(predict_controls, text='Clear Results', command=clear_results)
clear_results_btn.pack(side='left', padx=6)

log_frame = ttk.Frame(predict_tab)
log_frame.pack(fill='both', expand=True, padx=8, pady=6)

log_label = ttk.Label(log_frame, text='Application Log / Results', font=('Segoe UI', 12, 'bold'))
log_label.pack(anchor='w')

# Interactive results Treeview (replaces plain text log for predict outputs)
results_tree = ttk.Treeview(log_frame, columns=('idx','status','reason'), show='headings', height=12)
results_tree.heading('idx', text='Index')
results_tree.heading('status', text='Approval Status')
results_tree.heading('reason', text='Reject Reason')
results_tree.column('idx', width=80)
results_tree.column('status', width=180)
results_tree.column('reason', width=420)
results_tree.pack(fill='both', expand=True)

# configure tag styles for coloring
try:
    results_tree.tag_configure('approved', background='#001f00')
    results_tree.tag_configure('rejected', background='#2a0000')
    results_tree.tag_configure('unknown', background='#0b0b0b')
except Exception:
    pass

# attach double-click handler to show details
results_tree.bind('<Double-1>', on_result_double_click)

main.config(bg='SystemButtonFace')
main.mainloop()
