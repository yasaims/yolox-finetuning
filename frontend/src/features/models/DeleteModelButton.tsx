import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../../api/client";

interface Props {
  modelId: string;
  onDeleted?: () => void;
}

export default function DeleteModelButton({ modelId, onDeleted }: Props) {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const deletion = useMutation({
    mutationFn: () => api.deleteModel(modelId),
    onSuccess: async () => {
      queryClient.removeQueries({ queryKey: ["models", modelId], exact: true });
      onDeleted?.();
      await queryClient.invalidateQueries({ queryKey: ["models"] });
    },
    onSettled: () => setConfirming(false),
  });

  return (
    <span className="inline-actions">
      {confirming ? (
        <>
          <span>Delete this model?</span>
          <button type="button" className="danger" onClick={() => deletion.mutate()} disabled={deletion.isPending}>
            Confirm delete
          </button>
          <button type="button" onClick={() => setConfirming(false)} disabled={deletion.isPending}>
            Cancel
          </button>
        </>
      ) : (
        <button type="button" onClick={() => setConfirming(true)}>
          Delete
        </button>
      )}
      {deletion.error && <span className="error">{deletion.error.message}</span>}
    </span>
  );
}
