import { useParams } from "react-router-dom";

export default function ModelDetailPage() {
  const { id } = useParams();
  return <h2>Model {id}</h2>;
}
