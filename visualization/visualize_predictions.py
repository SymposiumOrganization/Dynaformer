import csem_exptrack
import numpy as np
from surrogate import  metrics
from surrogate.data_modules.data_modules import*
from os import listdir
from os.path import isfile, join
from surrogate.data_modules.data_modules import*
from surrogate.models import Dynaformer
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import numpy as np 
from yaml import safe_load
from tqdm import tqdm
from decimal import Decimal

def load_pickle(pred_dir, name):
    """
    Find the scalers in the current directory and loads them
    args:
        pred_dir: (Path) the directory where the scalers are stored
    """
    scalers = np.load(pred_dir / f'{name}.pkl', allow_pickle=True)
    return scalers

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def apply_scalers(dataset, scaler_c, scalers_v):
    """
    Apply the scalers to the data computed in compute_scalers
    args:
        scalers: the scalers computed in compute_scalers   
        data_dir: (Path) the data directory
    """
    voltage = dataset["voltage"]
    current = dataset["current"]
    scaled_voltage = [scalers_v.transform(np.array(voltage_prof).reshape(-1,1)).squeeze() for voltage_prof in voltage]
    scaled_current = [scaler_c.transform(np.array(current_prof.get_current_profile(20000)).reshape(-1,1)).squeeze() for current_prof in current]

    return {"voltage": scaled_voltage, "current": scaled_current}


    
def compute_metric_single(y_pred, y):
    rmse = metrics.compute_rmse(y_pred, y.squeeze().numpy())
    wasserstein = metrics.compute_wasserstein_distance(y_pred, y.squeeze().numpy())
    mse = metrics.compute_mse(y_pred, y.squeeze().numpy())
    return rmse, wasserstein, mse


def main():
    """
    Visualizer for data and predictions
    """
    st.title("Visualizer")
    is_model_selected = st.checkbox("Show model prediction")
    if is_model_selected:
        df_single = csem_exptrack.load_project("runs",["weights/dynaformer*"], logic="singlerun")
        df = df_single.sort_index(axis=1)      
        selected = st.selectbox("Select a run", sorted(df.columns,reverse=True),index=0)

    val_dir = st.text_input("Insert path to the dataset (Something like data/YYYY-MM-DD/HH-MM-SS/data", "data/variable_currents/2022-04-27/14-58-12/data")
    config_dataset = safe_load(open(Path(val_dir) / "../.hydra/config.yaml", 'r'))

    json_file = "metadata.json"
    prefix = "train"
    tr = "train_times"
    with open(Path(val_dir).joinpath(f'{json_file}'), 'r') as f:
        metadata = json.load(f)
    # Note that scalers are not computed on the test set.
    st.markdown("## Information regarding the dataset:")
    st.markdown(f"* Number of batches: {metadata[tr]//metadata['chunk_size']}")
    st.markdown(f"* Dimension of the batches: {config_dataset['chunk_size']}")
    batch = st.number_input("Select Test Batch", value=0, min_value=0, max_value=metadata[tr]//metadata["chunk_size"])
    batch = batch * config_dataset["chunk_size"]

    if is_model_selected:
        key = ("path",0)
        model_path = Path(df.loc[key,selected])
        st.write(model_path)
        model = Dynaformer.load_from_checkpoint(model_path)
        model.eval()

    metrics = {"rmse": [], "mse": [],  "wasserstein": []}

    current_path = Path(val_dir)/ f"{prefix}_currents_{batch}.pkl"
    voltage_path = Path(val_dir) / f"{prefix}_voltages_{batch}.pkl"
    time_path = Path(val_dir) / f"{prefix}_times_{batch}.pkl"
    with open(current_path, 'rb') as f:
        currents = pickle.load(f)
    with open(voltage_path, 'rb') as f:
        voltages = pickle.load(f)
    with open(time_path, 'rb') as f:
        times = pickle.load(f)

    dataset_len = len(currents)
    y_preds = []

    x_init = st.number_input('Initial value for context window',0,500,value=0)
    cutoff = st.number_input('cutoff',0,20000,value=3500) + x_init
    

    max_lengths = []
    for i in tqdm(range(dataset_len)):
        max_length = len(voltages[i])
        max_lengths.append(max_length)
        voltage = torch.tensor(voltages[i][:min(max_length,cutoff)]).unsqueeze(0)
        
        time = torch.tensor(times[i][:min(max_length,cutoff)]).unsqueeze(0)
        current = torch.tensor(currents[i].get_current_profile(voltage.shape[1]*2)).unsqueeze(0)
        current =  current[:, x_init:(max_length)]
       
        x = current.unsqueeze(2).float() # It already has x_init removed and up to max_length
        y = voltage.unsqueeze(2)[:min(max_length,cutoff)]
        t = time.unsqueeze(2)[:min(max_length,cutoff)]
        n=200
        xx,yy,tt=x[:,:n,:],y[:,x_init:(n+x_init),:], t[:,x_init:(n+x_init),:]
        inp = torch.cat([xx,yy,tt],axis=2).float()
        if is_model_selected:
            y_pred = model(inp, x.squeeze(2)[:,:]).cpu().detach().numpy().squeeze()
            y_pred = y_pred[:min(max_length,cutoff)]
            y_preds.append(y_pred)

            y = y[:,(x_init):]
            
            rmse, wasserstein, mse = compute_metric_single(y_pred, y.squeeze())
            metrics["rmse"].append(rmse)
            metrics["wasserstein"].append(wasserstein)
            metrics["mse"].append(mse)    


    # Create a figure with plotly
    fig = go.Figure()
    n=st.number_input('Select curve within the batch: ', min_value=0, max_value=int(dataset_len))
    Q=np.load(Path(val_dir) / f'{prefix}_Qs_{str(batch)}.pkl', allow_pickle=True)
    R=np.load(Path(val_dir) / f'{prefix}_Rs_{str(batch)}.pkl', allow_pickle=True)
    q_=Q[n]
    r_=R[n]
    current = currents[n].get_current_profile(40000)

    gt =voltages[n][:cutoff]
    
    # Print number of transitions of the current
    st.write(f"Number of transitions: {len(set(current[:cutoff]))-1}")
    st.write(f"Transitions: {(set(current[:cutoff]))}")
    st.write(f"Transition Steps: {2*(np.where(np.diff(current[:cutoff])!=0))[0] + 2*x_init}")
    
    max_length = max_lengths[n]
    xx_g = np.arange(0,2*len(gt[:min(max_length,cutoff)]),2) #FIXME
    if is_model_selected:
        pred = y_preds[n]
        xx_p = np.arange(0,2*len(pred),2) + x_init*2
        fig.add_trace(go.Scatter(x = xx_p,y=pred, mode='markers', name='Predicted'))
    fig.add_trace(go.Scatter(x = xx_g,y=gt, mode='lines', name='Ground Truth',line=dict(color='firebrick', width=4,
                            dash='dash')))
    # Add a vertical line where pred is equal to 3.2 V
    cutoff_value = 3.2
    # Find the index closest to the cutoff value
    cutoff_index = np.argmin(np.abs(gt - cutoff_value))

    is_vertical_line = st.checkbox("Add vertical line at 3.2 for the GT", value=False)
    if is_vertical_line:
        # Add a vertical line at the cutoff index
        fig.add_vline(x=xx_p[cutoff_index])

    is_context_highlighted = st.checkbox("Add highligth for the context", value=True)
    if is_context_highlighted:
        
        # Highlight the context region
        fig.add_vrect(
            x0=x_init*2, x1=(x_init+200)*2,
            fillcolor="black", opacity=0.2,
            layer="below", line_width=0,
        )


    fig.update_layout(
        xaxis=dict(
        mirror=True, showline=True,linecolor = "black",),
        yaxis=dict(
        mirror=True, showline=True,linecolor = "black",),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        title={
        'y':0.9,
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'}
    )
    fig.update_layout(
        title_font_family="sans-serif",
        font_size=18
    )
    fig.update_xaxes(nticks=10,ticks="outside", tickwidth=2)
    fig.update_yaxes(nticks=10,ticks="outside", tickwidth=2)
    fig.update_layout(xaxis=dict(showgrid=False),
              yaxis=dict(showgrid=False)
            )

    fig.update_layout(showlegend=False)
    fig.update_layout(
        title_text="Q=" + str(np.round(q_,2)) + " " + "R=" + str(np.round(r_,2)) + " " + "I=" + str(np.round(current[0],2))
    )
    st.write(fig)

    if is_model_selected:
        st.write(f"**Local Metrics**")
        st.write(f"RMSE: {metrics['rmse'][n]}")
        st.write(f"Wasserstein: {metrics['wasserstein'][n]}")
        st.write(f"MSE: {metrics['mse'][n]}")



if __name__ == '__main__':
    main()