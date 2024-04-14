job_ids = ['abe47d66-d0e0-4ba1-87a6-d78ced79eb23', '6ae599b3-666e-43b1-bcaf-13d329781ba5','5ec735a2-a906-4ca6-9f45-a4a3403c9a1b','10cf8eb1-0d29-441a-9fc3-10845ef3c7e7','f4df51ca-9afc-4151-bee7-95ed1a1d4467','9bfea2f7-b812-4854-8ba3-9a33ef873e12','53a7fb5c-eaf0-405a-884d-722119b05153','41c5f526-37b0-406c-a728-848bd6c998dc','eb5bb357-ccae-4dd7-bf52-836943466c49','cac2ac11-cf7c-42b5-a556-4440096d7a80','3f125cb1-3c57-47b1-abcc-4396067d0098','83466f64-7fc5-49b2-9008-8e53697f6589','c17c8ae3-8816-49dc-ac4d-f8a6eed84345','df0ec160-ab63-4b34-a3cd-e7b6af3ae38b','a4db40d3-70f9-4c86-81b0-7fe9667343ae','63aab7f3-1f93-45b5-8919-fc2cfb7bb69f','9b777af1-071b-4a67-a829-236dced29bf2','8c512560-4bb0-4aef-86b7-9daac3595312']

from qiskit_ionq import IonQProvider
from qiskit_ionq.ionq_job import IonQJob

ionq_provider = IonQProvider('TwgR8FPezAeiQbxM5nYwuqGLk0b8Ynps')

backend = ionq_provider.get_backend("ionq_qpu.aria1")
job = IonQJob(backend=backend, job_id='abe47d66-d0e0-4ba1-87a6-d78ced79eb23')
counts = job.get_counts()
print(counts)
hybrid_results = []
enhanced_results = []
for i, job_id in enumerate(job_ids):
    if i % 2 == 0:
        job = IonQJob(backend=backend, job_id=job_id)
        counts = job.get_counts()
        hybrid_results.append(counts)
    if i % 2 == 1:
        job = IonQJob(backend=backend, job_id=job_id)
        counts = job.get_counts()
        enhanced_results.append(counts)
print(hybrid_results)
print(enhanced_results)

