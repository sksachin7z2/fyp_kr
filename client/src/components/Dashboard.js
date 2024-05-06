import React,{useState,useEffect} from 'react'
import DragDropFile from './DragandDrop'
import ProgressBar from './Loader'
import axios from 'axios'
import BarChart from './Chart'
function Dashboard() {
    const [files, setFiles] = useState([])
    const [load, setLoad] = useState(0)
    const [pred, setPred] = useState("")
    const [time, setTime] = useState([])
    const [weigh, setWeigh] = useState([])

    useEffect(() => {
      let ui,oi
      if(localStorage.getItem('prob')){
      ui= JSON.parse( localStorage.getItem('prob'))
      oi=(JSON.parse(ui))
      }
      else{
        oi={}
      }

      const keysArray = Object.keys(oi);
      const valuesArray = Object.values(oi);
      setTime(keysArray)
      setWeigh(valuesArray)
    }, [])
    
    let chartData={
      labels:time?.map((data)=>data),
      
      datasets:[
        {
          data:weigh?.map((data)=>data),
          label:"prediction"
        },
      
      ],
    }
    const sendAudioToBackend = () => {
        const formData = new FormData();
        let audioBlob=document.getElementById('input-file-upload').files[0]
        // console.log(audioBlob)
        formData.append('file', audioBlob);
        setLoad(50)
        axios.post('http://127.0.0.1:5000/predict', formData,{
          headers:{
            'Content-Type': 'multipart/form-data',
          }
        })
          .then((response) => {
            console.log(response.data)
            // setPred(response.data['pred'])
            if(!response.data['error'])
            {
            localStorage.setItem('pred',response.data['pred'])
            localStorage.setItem('prob',JSON.stringify(response.data['prob']))
            setLoad(100)
            window.location.reload()
            
            }
            else{
            // setPred(localStorage.getItem('pred'))
            setLoad(100)
            alert(response.data['error'])
            window.location.reload()
            }
          })
          .catch((error) => {
            console.error('Error:', error);
          });
      };
  return (
    <div className='w-[90vw] m-auto '>
        <div onClick={()=>{console.log(files)}} className='text-center my-6 text-[2.5rem] font-semibold'>
        Cancer Detection System
        </div>
        <DragDropFile files={files} setFiles={setFiles}/>

        <div className='w-50 my-5' >

        <div className='my-3 text-center'>
      <button className='px-2 py-1 text-md font-semibold bg-blue-600 text-white rounded-md' onClick={sendAudioToBackend}>
        Predict
      </button>
      </div>
<ProgressBar progressPercentage={load}/>
  </div>
        <section className='my-5'>
            <div className='flex gap-[4rem] justify-center items-center'>
            <div className='text-[1.5rem] font-semibold'>
                Prediction
            </div>
            <div className='text-[1.5rem] font-semibold text-green-500'>
                  {!localStorage.getItem('pred')?"No input given":localStorage.getItem('pred')}
            </div>
            </div>
           
        </section>
        <section className='mb-[3rem]'>
        <div className='font-semibold text-center text-[2rem] mt-[4rem] mb-[2rem]'>Analytics</div>
        <div className='w-[50vw] m-auto'>

        <BarChart chartData={chartData}/>
        </div>

        </section>

    </div>
  )
}

export default Dashboard