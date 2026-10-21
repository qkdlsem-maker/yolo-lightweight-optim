"""Observe native optimizer execution without changing its update or EMA rules."""
class OptimizerStepAccounting:
    def install_step_accounting(self, restored=None):
        assert not hasattr(self,'_actual_step_hook')
        self.actual_step_counts=dict(restored or {'attempted':0,'applied':0,'skipped':0})
        assert self.actual_step_counts['attempted']==self.actual_step_counts['applied']+self.actual_step_counts['skipped']
        def applied(optimizer,args,kwargs): self.actual_step_counts['applied']+=1
        self._actual_step_hook=self.optimizer.register_step_post_hook(applied)

    def optimizer_step(self):
        before=self.actual_step_counts['applied']
        self.actual_step_counts['attempted']+=1
        super().optimizer_step()
        applied=self.actual_step_counts['applied']-before
        assert applied in [0,1]
        self.actual_step_counts['skipped']+=1-applied
        assert self.actual_step_counts['attempted']==self.actual_step_counts['applied']+self.actual_step_counts['skipped']
