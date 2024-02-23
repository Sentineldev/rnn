import { useState } from 'react';
import './App.css'
import Predict from './components/Predict';
import Train from './components/Train';



export type Rnn = {

  input_weights: number[][]
  hidden_weights: number[][]
  hidden_bias: number[][]
  output_weights: number[][]
  output_bias: number[][]
  hidden_units: number;
}

export default function App() {



  const [rnn, setRnn] = useState<Rnn>()

  return (
    <div className="lg:grid grid-cols-2 h-full">
      <div className="col-span-1">
        <Train SetRnn={setRnn}/>
      </div>
      { rnn && (
        <div className="col-span-1">
          <Predict rnn={rnn}/>
        </div>

      ) }
    </div>
  );
}