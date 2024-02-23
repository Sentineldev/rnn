import { useState } from "react";
import FormInput from "./FormInput";
import { Loader } from "./Loader";
import { useForm } from "react-hook-form";
import Alert from "./Alert";
import { Rnn } from "../App";


export type FormDataBody = {
    epochs: number;
    hiddens: number;
}

export type TrainProps = {
    SetRnn: (rnn: Rnn | undefined) => void;
}

export default function Train({ SetRnn }: TrainProps) {


    const [onLoading, setLoading] = useState<boolean>(false);

    const [logs, setLogs] = useState<string[]>([]);

    const {  register, handleSubmit  } = useForm()

    const [responseMessage, setResponseMessage] = useState<string>("");
    async function onSubmitHandler(formDataBody: FormDataBody) {
        setResponseMessage("");
        setLogs([]);
        setLoading(true);
        SetRnn(undefined);
        try {
            const response = await fetch("http://127.0.0.1:3000/train", { 
                method: "POST",
                body: JSON.stringify(formDataBody)
             });
    
            if (response.status === 200) {
                const data = await response.json()
                setLogs(data.logs)
                data.weights as Rnn
                SetRnn({
                    ...data.weights,
                    hidden_units: formDataBody.hiddens
                })
                setResponseMessage("Training finish! 😃")
            }
            setLoading(false)
        } catch (error) {
            setLoading(false);
            setResponseMessage("Ocurrio un error entrenando la red. 😰")
        }  

    }

    return (
        <>
        <div className="p-12">
            <header className="font-bold text-xl">Parametros de entrenamiento</header>
            <form onSubmit={handleSubmit((data) => onSubmitHandler(data as FormDataBody))} className="">
                <div className="py-4">
                    <FormInput label="Epocas" props={{ type: "number", ...register("epochs", { required: true, valueAsNumber: true }) }}/>
                    <FormInput label="Numero de Neuronas" props={{ type: "number", ...register("hiddens", { required: true, valueAsNumber: true }) }}/>
                </div>
                <button disabled={onLoading} className="btn btn-primary border-none">Entrenar</button>
            </form>
            { responseMessage.length > 0 
            && 
            <div className="py-4">
                <Alert message={responseMessage} />
            </div>
            }
            { onLoading &&
            <div className="py-4">
                <Loader color="primary" size="lg"/>
            </div>
            }
        </div>
        { logs.length > 0 && (
            <div className="px-12">
                <header className="font-bold text-2xl">Logs de entrenamiento</header>
                { logs.map((log, index) => (
                    <>
                    { index == logs.length - 1  ? 
                    <p key={`some-p-${index}`}>{log}</p>
                    :
                    <p key={`some-p-${index}`} className="border-b py-2">{log}</p>
                    } 
                    </> 
                )) }
            </div>

        ) }
        </>
    );
}

