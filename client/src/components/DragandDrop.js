import React,{useState} from 'react'

export default function DragDropFile({setFiles,files}) {
 
    // drag state
    const [dragActive, setDragActive] = React.useState(false);
    // ref
   
    const inputRef = React.useRef(null);
    
    function handleFile(file) {
        let arr=[]
        let arr1=[]
       Array.from(file).map((e,i)=>{
              arr1.push(e);
            var fileName = URL.createObjectURL(file[i]);
            arr.push(fileName)
        })
        // setPhotos([...photos,...arr])
        setFiles([...files,...arr1])
        console.log(arr1)
       
      }
    // handle drag events
    const handleDrag = function(e) {
      e.preventDefault();
      e.stopPropagation();
      if (e.type === "dragenter" || e.type === "dragover") {
        setDragActive(true);
      } else if (e.type === "dragleave") {
        setDragActive(false);
      }
    };
    
    // triggers when file is dropped
    const handleDrop = function(e) {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files);
      }
    };
    
    // triggers when file is selected with click
    const handleChange = function(e) {
      e.preventDefault();
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files);
      }
    };
    
  // triggers the input when the button is clicked
    const onButtonClick = () => {
      inputRef.current.click();
    };
    // const handledelete=(id)=>{
    //     let arr=photos;
    //     delete arr[id];
    //     let ans=arr.filter(e=>e!==undefined)
    //     setPhotos(ans)

    //     let arr1=files;
    //     delete arr1[id];
    //     let ans1=arr1.filter(e=>e!==undefined)
    //     setFiles(ans1)

    // }
    return (
        <div>
<form className='m-auto' id="form-file-upload" onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
        <input ref={inputRef} type="file" id="input-file-upload" onChange={handleChange} />
        <label id="label-file-upload" htmlFor="input-file-upload" className={dragActive ? "drag-active" : "" }>
      
          <div >
          <svg className='m-auto' fill="#000000" version="1.1" xmlns="http://www.w3.org/2000/svg" width="73px" height="73px" viewBox="0 0 31.06 32.001" ><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <g id="photos"> <path d="M29.341,11.405L13.213,7.383c-1.21-0.301-2.447,0.441-2.748,1.652L6.443,25.163c-0.303,1.211,0.44,2.445,1.65,2.748 l16.127,4.023c1.21,0.301,2.447-0.443,2.748-1.652l4.023-16.127C31.293,12.944,30.551,11.708,29.341,11.405z M28.609,14.338 l-2.926,11.731c-0.1,0.402-0.513,0.65-0.915,0.549l-14.662-3.656c-0.403-0.1-0.651-0.512-0.551-0.916l2.926-11.729 c0.1-0.404,0.513-0.65,0.916-0.551l14.661,3.658C28.462,13.522,28.71,13.936,28.609,14.338z"></path> <circle cx="15.926" cy="13.832" r="2.052"></circle> <path d="M22.253,16.813c-0.136-0.418-0.505-0.51-0.82-0.205l-2.943,2.842c-0.315,0.303-0.759,0.244-0.985-0.133l-0.471-0.781 c-0.227-0.377-0.719-0.5-1.095-0.273l-4.782,2.852c-0.377,0.225-0.329,0.469,0.096,0.576l3.099,0.771 c0.426,0.107,1.122,0.281,1.549,0.389l3.661,0.912c0.426,0.105,1.123,0.279,1.549,0.385l3.098,0.773 c0.426,0.107,0.657-0.121,0.521-0.539L22.253,16.813z"></path> <path d="M2.971,7.978l14.098-5.439c0.388-0.149,0.828,0.045,0.977,0.432l1.506,3.933l2.686,0.67l-2.348-6.122 c-0.449-1.163-1.768-1.748-2.931-1.299L1.45,6.133C0.287,6.583-0.298,7.902,0.151,9.065L5.156,22.06l0.954-3.827L2.537,8.954 C2.389,8.565,2.583,8.126,2.971,7.978z"></path> </g> <g id="Layer_1"> </g> </g></svg>

            <p>Drag and drop your file here or</p>
            <button className="upload-button" onClick={onButtonClick}>Upload a file</button>
          </div> 
        </label>
        { dragActive && <div id="drag-file-element" onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}></div> }
      </form>

        </div>
      
    );
  };