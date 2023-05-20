
def simulate_model(
    instance: torch.FloatTensor, events_scores: torch.FloatTensor) -> torch.FloatTensor:
    #eps = torch.rand_like(events_scores).float().to(instance.device)
    #eps_hat = (eps.log() - (1 - eps).log() + instance) / 1.0
    events_scores = events_scores.sigmoid()
    #eps = torch.rand_like(events_scores)
    eps = torch.rand(1).to(instance.device)
    events_scores = torch.sigmoid((eps.log() - (1 - eps).log() + events_scores) / 2.0)
    # TODO: Simulate all events, not just the speed events.
    #relaxed_bernoulli = torch.distributions.RelaxedBernoulli(2.0, events_scores)
    #e = torch.sigmoid(events_scores)
    #e = relaxed_bernoulli.sample()
    #e = torch.rand(1).float().to(instance.device)
    #e_hat = torch.sigmoid((torch.log(e) - torch.log(1 - e) + events_scores) / .05)
    result = events_scores >= .5
    #print(result)
    #result = torch.bernoulli(events_scores)
    instance = result * instance
    return instance