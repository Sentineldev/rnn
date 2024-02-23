import { useState } from "react";
import { Rnn } from "../App";
import { Loader } from "./Loader";

export type PredictProps = {
    rnn: Rnn

}
export type PredictCase = {
    date: string;
    t_max: number;
    t_min: number;
    rain: number; //humidity
    t_tomorrow: number;
}
// 2022-10-21,64.0,53.0,0.0,69.0
// 2022-10-22,69.0,57.0,0.0,70.0
// 2022-10-23,70.0,49.0,0.0,72.0
// 2022-10-24,72.0,44.0,0.0,66.0
// 2022-10-25,66.0,48.0,0.0,66.0
// 2022-10-26,66.0,45.0,0.0,69.0
// 2022-10-27,69.0,43.0,0.0,65.0
//case2
// 2022-08-01,75.0,60.0,0.0,76.0
// 2022-08-02,76.0,59.0,0.0,72.0
// 2022-08-03,72.0,58.0,0.0,74.0
// 2022-08-04,74.0,51.0,0.0,74.0
// 2022-08-05,74.0,62.0,0.0,73.0
// 2022-08-06,73.0,61.0,0.0,73.0
// 2022-08-07,73.0,61.0,0.0,74.0
//case objects
// const cases: Array<PredictCase> = [
//     { date: '2022-10-21',t_max: 64.0, t_min: 63.0, rain: 0.0, t_tomorrow: 69.0 },
//     { date: '2022-10-22',t_max: 69.0, t_min: 57.0, rain: 0.0, t_tomorrow: 70.0 },
//     { date: '2022-10-23',t_max: 70.0, t_min: 49.0, rain: 0.0, t_tomorrow: 72.0 },
//     { date: '2022-10-24',t_max: 72.0, t_min: 44.0, rain: 0.0, t_tomorrow: 66.0 },
//     { date: '2022-10-25',t_max: 66.0, t_min: 48.0, rain: 0.0, t_tomorrow: 66.0 },
//     { date: '2022-10-26',t_max: 66.0, t_min: 45.0, rain: 0.0, t_tomorrow: 69.0 },
//     { date: '2022-10-27',t_max: 69.0, t_min: 43.0, rain: 0.0, t_tomorrow: 65.0 },
// ] 

const cases: Array<PredictCase> = [
    { date: '2022-08-01',t_max: 75.0, t_min: 60.0, rain: 0.0, t_tomorrow: 76.0 },
    { date: '2022-08-02',t_max: 76.0, t_min: 59.0, rain: 0.0, t_tomorrow: 72.0 },
    { date: '2022-08-03',t_max: 72.0, t_min: 58.0, rain: 0.0, t_tomorrow: 74.0 },
    { date: '2022-08-04',t_max: 74.0, t_min: 51.0, rain: 0.0, t_tomorrow: 74.0 },
    { date: '2022-08-05',t_max: 74.0, t_min: 62.0, rain: 0.0, t_tomorrow: 73.0 },
    { date: '2022-08-06',t_max: 73.0, t_min: 61.0, rain: 0.0, t_tomorrow: 73.0 },
    { date: '2022-08-07',t_max: 73.0, t_min: 61.0, rain: 0.0, t_tomorrow: 74.0 },
] 

export default function Predict({ rnn }: PredictProps) {    


    const [selectedCases, setCases] = useState<Array<PredictCase>>([]);

    const [predicted, setPredicted] = useState<Array<number>>([]);
    const [onLoading, setOnLoading] = useState(false);

    function onSelectHandler(x: PredictCase, action: "ADD" | "REMOVE") {
        if (action === "ADD") {
            if (selectedCases.length === 7) {
                alert("Numero de casos maximo alcanzado.")
                return
            }
            setCases((previous) => [...previous,x])
        }
        if (action === "REMOVE") {
            setCases((previous) => previous.filter((y) =>  y.date !== x.date))
        }
    }

    function CheckIfCaseInList(x: PredictCase) {
        return selectedCases.find((y) => x.date === y.date)
    }

    async function PredictHandler() {

        setOnLoading(false);
        setPredicted([]);
        if (selectedCases.length !== 7) {

            alert("Debe seleccionar al menos 7 casos.")
            return;
        }   
        setOnLoading(true);
        try {
            const response = await fetch("http://127.0.0.1:3000/predict", {
                method: "POST",
                body: JSON.stringify({
                    Data: selectedCases,
                    Rnn: rnn
                })
            })
            const data = await response.json()
            setPredicted(data.outputs)
            setOnLoading(false);
        } catch (error) {
            setOnLoading(false);
        }
        
    }

    return (
        <>
        <div className="p-12">
            <header className="font-bold text-xl">Realizar Prediccion</header>
            
            <div className="overflow-x-auto">
                <table className="table">
                    {/* head */}
                    <thead>
                        <tr>
                            <th>Fecha</th>
                            <th>TMax</th>
                            <th>TMin</th>
                            <th>Humedad</th>
                            <th>T Tomorrow</th>
                        </tr>
                    </thead>
                    <tbody>
                    { cases.map((x) => (
                        <>
                        { CheckIfCaseInList(x) ? (
                            <tr onClick={() => { onSelectHandler(x, "REMOVE") }} className="cursor-pointer bg-neutral-600">
                                <th>{x.date}</th>
                                <th>{x.t_max}</th>
                                <td>{x.t_min}</td>
                                <td>{x.rain}</td>
                                <td>{x.t_tomorrow}</td>
                            </tr>
                        ) : (
                            <tr onClick={() => { onSelectHandler(x, "ADD") }} className="cursor-pointer">
                                <th>{x.date}</th>
                                <th>{x.t_max}</th>
                                <td>{x.t_min}</td>
                                <td>{x.rain}</td>
                                <td>{x.t_tomorrow}</td>
                            </tr>
                        ) }
                        </>
                        
                    )) }
                    </tbody>
                </table>
            </div>
            <div className="py-4">
                <button disabled={onLoading} onClick={PredictHandler} className="btn btn-primary border-none">Generar Prediccion</button>
            </div>
            { onLoading &&
            <div className="py-4">
                <Loader color="primary" size="lg"/>
            </div>
            }
            { predicted.length > 0 && (
                <table className="table">
                    {/* head */}
                    <thead>
                        <tr>
                            <th>Prediccion</th>
                        </tr>
                    </thead>
                    <tbody>
                    { predicted.map((x) => (
                        <>
                        <tr className="cursor-pointer">
                            <th>{x}</th>
                        </tr>
                        </>
                        
                    )) }
                    </tbody>
                </table>

            ) }
        </div>
        </>
    );
}