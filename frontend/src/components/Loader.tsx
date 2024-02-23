export type LoaderProps = {
    size: "xs" | "sm" | "md" | "lg"
    color: string;
}
export function Loader({ size, color }: LoaderProps) {

    return (
        <span className={`loading loading-spinner loading-${size} text-${color}`}></span>
    );
}